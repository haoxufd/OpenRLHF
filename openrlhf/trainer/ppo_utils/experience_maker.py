import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import json

import re

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward_new, masked_mean

from openrlhf.utils.utils import convert_token_to_id
from xuhao.utils import get_steps
from xuhao.utils import get_final_value_from_solution
from xuhao.utils import group_elements
from xuhao.utils import solution_end_is_valid
from xuhao.utils import get_eostep_indices

from openrlhf.utils.utils import convert_token_to_id


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self

@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    ref_solutions: list[str]

    def subset(self, indices: list[int]) -> "Samples":
        # 提取指定索引的样本
        sequences = self.sequences[indices]
        attention_mask = self.attention_mask[indices] if self.attention_mask is not None else None
        action_mask = self.action_mask[indices] if self.action_mask is not None else None
        num_actions = self.num_actions[indices] if isinstance(self.num_actions, torch.Tensor) else self.num_actions
        packed_seq_lens = self.packed_seq_lens[indices] if self.packed_seq_lens is not None else None
        response_length = self.response_length[indices]
        total_length = self.total_length[indices]
        ref_solutions = [self.ref_solutions[i] for i in indices]

        # 返回新的 Samples 实例
        return Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            response_length=response_length,
            total_length=total_length,
            ref_solutions=ref_solutions
        )


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
        logger=None
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.verification_system_message_file = "xuhao/verify/data/input/verification_system_message.txt"
        self.verification_few_shot_file = "xuhao/verify/data/input/verification_few_shot.json"
        self.logger = logger

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str], List[List[str]]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # generate responses
        samples_list = self.generate_samples(all_prompts, **generate_kwargs)

        num_samples = len(samples_list)
        self.logger.info(f"There are totally {num_samples} 'Samples', is going to make {num_samples} 'Experience'")

        experiences = []
        cnt = 0
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            self.logger.info(f"Make 'Experience' {cnt}......")
            experience = self.make_experience(samples)
            experiences.append(experience.to_device("cpu"))
            cnt += 1

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        idx = 0
        step_split_token_id = convert_token_to_id(args.step_split_str, self.tokenizer)
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            num_actions = experience.info["num_actions"]
            sequences = experience.sequences
            seq_len = sequences.size(1)
            response_sequences = sequences[:, (seq_len - num_actions):].tolist()
            assert sequences[:, (seq_len - num_actions):].shape == experience.kl.shape

            # for i in range(len(response_sequences)):
            #     # 判断 response 是以 <|eot_id|><|end_of_text|> 结尾还是以 <|reserved_special_token_0|><|end_of_text|> 结尾
            #     # 如果以 <|eot_id|><|end_of_text|> 结尾, reward 额外 +1
            #     first_end_of_text_idx = response_sequences[i].index(convert_token_to_id("<|end_of_text|>", self.tokenizer))
            #     if response_sequences[i][first_end_of_text_idx - 1] == convert_token_to_id("<|eot_id|>", self.tokenizer):
            #         reward[i][-1] += 1
            
            # 更新 experience.info["reward"]
            average_rewards = [sum(sublist) / len(sublist) for sublist in reward]
            average_rewards_tensor = torch.tensor(average_rewards, dtype=torch.float32)
            experience.info["reward"] = average_rewards_tensor

            eostep_indices = get_eostep_indices(response_sequences, step_split_token_id)
            self.logger.info("End of Step Indices:")
            self.logger.info(eostep_indices)
            self.logger.info("Reward of Current Experience:")
            self.logger.info(reward)
            assert len(eostep_indices) == len(reward)
            for i in range(len(eostep_indices)):
                assert len(eostep_indices[i]) == len(reward[i]) or len(eostep_indices[i]) == (len(reward[i]) + 1)
                if len(eostep_indices[i]) == (len(reward[i]) + 1):
                    assert eostep_indices[i][-1] ==  (eostep_indices[i][-2] + 1)
                    eostep_indices[i].pop(-1)
            reward = compute_reward_new(
                reward,
                eostep_indices,
                self.kl_ctl.value,
                experience.kl,
                experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
            idx += 1
        return experiences
    
    def is_valid(self, solution: str):
        steps = solution.split('\n')
        last_step = steps[-1].strip()
        if not (re.match(r"^####\s+-?[\d,]+(\.[\d,]+)?$", last_step) or re.match(r"^####\s+None$", last_step)):
            return False
        
        return True

    def get_problem_and_solution_qwen(self, text: str):
        special_token_content = {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>",
            "text_start": "<|beginoftext|>",
            "text_end": "<|endoftext|>"
        }
        tmp = text.split(special_token_content["im_start"]+"user")[-1]
        problem = tmp.split(special_token_content["im_end"])[0].strip()
        sol_start_content = special_token_content["im_start"] + "assistant"
        sol_end_content = special_token_content["im_end"]
        sol_start_idx = tmp.find(sol_start_content) + len(sol_start_content)
        sol_end_idx = tmp.find(sol_end_content, sol_start_idx + len(sol_start_content))
        solution = tmp[sol_start_idx: sol_end_idx].strip()

        return problem, solution

    def get_problem_and_solution_llama(self, text: str):
        header_start = "<|start_header_id|>"
        header_end = "<|end_header_id|>"
        text_start = "<|begin_of_text|>"
        text_end = "<|end_of_text|>"
        round_end = "<|eot_id|>"

        user_split = header_start + "user" + header_end
        assistant_split = header_start + "assistant" + header_end
        tmp = text.split(user_split)[-1]
        problem = tmp[:tmp.find(round_end)].strip()
        tmp = text.split(assistant_split)[-1]
        solution = tmp[:tmp.find(text_end)].strip()

        return problem, solution

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[List[str]], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        self.logger.info("Generating 'Samples'......")
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_problems = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts[0]], [])
        all_ref_solutions = sum([[ref_sol] * args.n_samples_per_prompt for ref_sol in all_prompts[1]], [])
        samples_list = []
        for i in range(0, len(all_problems), args.micro_rollout_batch_size):
            self.logger.info(f"Micro rollout batch {i / args.micro_rollout_batch_size}")
            prompts = all_problems[i : i + args.micro_rollout_batch_size]
            ref_solutions = all_ref_solutions[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device=torch.cuda.current_device())

            while True:
                self.logger.info(f"Generate kwargs: {generate_kwargs}")
                sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
                
                texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)
                
                tmp = [self.get_problem_and_solution_llama(text) for text in texts]
                problems = [x[0] for x in tmp]
                solutions = [x[1] for x in tmp]
                all_valid = all([solution_end_is_valid(solution) for solution in solutions])

                if all_valid:
                    break
                else:
                    self.logger.info(f"Problems are {problems}")
                    self.logger.info(f"Solutions are {solutions}")
                    self.logger.info("There are invalid solutions, regenerating......")

            if sequences.size(0) > 0:
                samples = Samples(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    num_actions=action_mask.size(1),
                    packed_seq_lens=None,
                    response_length=action_mask.float().sum(dim=-1),
                    total_length=attention_mask.float().sum(dim=-1),
                    ref_solutions=ref_solutions
                )
                samples_list.append(samples)

        return samples_list
    
    def preprocess_data(self, problem, ref_solution, previous_steps, step) -> str:

        # Add No. to previous_steps and step
        num_previous_steps = len(previous_steps)
        previous_steps = [f"{idx+1}. " + step for idx, step in enumerate(previous_steps)]
        previous_steps = '\n'.join(previous_steps)
        step = f"{num_previous_steps + 1}. " + step
        data = f"Problem:\n{problem}\n\nReference Solution:\n{ref_solution}\n\nPrevious Steps:\n{previous_steps}\n\nStep to Evaluate:\n{step}"

        with open(self.verification_system_message_file, 'r') as f:
            system_message = f.read()

        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": data})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def verify(self, samples: Samples):
        self.logger.info("Verify solutions by step with reward model......")
        texts = self.tokenizer.batch_decode(samples.sequences, skip_special_tokens=False)
        problems = []
        solutions = []
        ref_solutions = samples.ref_solutions
        final_values = []
        ref_values = []
        for text in texts:
            problem, solution = self.get_problem_and_solution_llama(text)
            problems.append(problem)
            solutions.append(solution)
            final_values.append(get_final_value_from_solution(solution))
        
        for ref_solution in ref_solutions:
            ref_values.append(get_final_value_from_solution(ref_solution))

        solution_labels = [True if final_values[i] == ref_values[i] else False for i in range(len(final_values))]

        self.logger.info("Problems>>>>>>")
        self.logger.info(problems)
        self.logger.info("Solutions>>>>>>")
        self.logger.info(solutions)
        self.logger.info("Reference solutions>>>>>>")
        self.logger.info(ref_solutions)
        self.logger.info("Solution labels>>>>>>")
        self.logger.info(solution_labels)

        # Get steps for solutions
        step_list = []
        for solution in solutions:
            step_list.append(get_steps(solution))
        
        self.logger.info("Steps of each solution>>>>>>")
        for idx, steps in enumerate(step_list):
            self.logger.info(f"Steps for solution {idx}")
            self.logger.info(steps)
        
        prompts = []
        for idx, problem in enumerate(problems):
            for idy, step in enumerate(step_list[idx]):
                prompts.append(self.preprocess_data(problem, ref_solutions[idx], step_list[idx][:idy], step))
        
        assert self.strategy is not None
        micro_prompt_list = []
        batch_size = self.strategy.args.reward_model_generate_batch_size
        prompt_max_len = self.strategy.args.rm_prompt_max_len
        for i in range(0, len(prompts), batch_size):
            micro_prompt_list.append(prompts[i : i + batch_size])
        
        outputs = []
        for micro_promts in micro_prompt_list:
            micro_inputs = self.tokenize_fn(micro_promts, prompt_max_len, device=torch.cuda.current_device())
            micro_outputs = self.reward_model.model.generate(
                **micro_inputs,
                use_cache=True,
                max_new_tokens=512,
                do_sample=True,
                top_p=1.0,
                early_stopping=False,
                num_beams=2,
                temperature=1.0,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            input_length = micro_inputs["input_ids"].shape[1]
            micro_outputs = self.tokenizer.batch_decode(micro_outputs[:, input_length:], skip_special_tokens=True)
            outputs.extend(micro_outputs)
        
        r = [False if "Evaluation Result: INCORRECT" in output else True for output in outputs]
        num_step = [len(steps) for steps in step_list]

        assert sum(num_step) == len(r)
        assert sum(num_step) == len(outputs)

        outputs = group_elements(outputs, num_step)
        r = group_elements(r, num_step)

        self.logger.info("Verification result for all solutions>>>>>>")
        for idx, output in enumerate(outputs):
            self.logger.info(f"Verification result for solution {idx}>>>>>>")
            for idy, data in enumerate(output):
                self.logger.info(f"Step {idy}>>>>>>")
                self.logger.info(data)

        for reward in r:
            if False in reward:
                index = reward.index(False)
                reward[index + 1:] = [False] * (len(reward) - index - 1)
        
        self.logger.info("Verification result accumulation>>>>>>")
        self.logger.info(r)
        
        picked_items = []
        for i in range(len(solution_labels)):
            if solution_labels[i] == all(r[i]):
                picked_items.append(i)
        
        self.logger.info("Picked items>>>>>>")
        self.logger.info(picked_items)

        if picked_items == []:
            picked_items = [0]

        return r, picked_items

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience|None:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        verification_result, picked_items = self.verify(samples)
        verification_result = [verification_result[i] for i in picked_items]
        assert self.strategy is not None
        r = [[self.strategy.args.correct_step_reward if x else self.strategy.args.incorrect_step_reward for x in res] for res in verification_result]
        
        samples = samples.subset(picked_items)

        assert len(verification_result) == samples.sequences.shape[0]

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
    
    def find_solution_end_index(self, solution: str):
        return solution.find("<|im_end|>") - 1
    
    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[List[List[float]]]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns