from transformers import AutoTokenizer
from xuhao.utils import find_newline_indices
import torch

from openrlhf.models.utils import compute_reward_new

from xuhao.utils import find_numbers, solution_end_is_valid

def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/user/models/Qwen2.5-1.5B-Instruct", 
        trust_remote_code=True, 
        use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    texts1 = [
        "Hi, there!\n\nNice to meet you!\n",
        "Hi, there!",
        "Hi, there!\n\nNice to meet",
        "Hi, there!\n\nNice to",
        "Hi, there!\n\nNice",
        "Hi, there!\n\n<|endoftext|>"
    ]
    batch1 = tokenizer(
        texts1,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=512,
        padding=True,
        truncation=True,
    )
    tmp = tokenizer.batch_encode_plus(texts1)
    print(tmp)

    texts2 = tokenizer.batch_decode(batch1["input_ids"], skip_special_tokens=False)

    print(batch1)
    print(texts2)

    print(tokenizer.batch_decode([[13048, 11, 1052, 2219]]))

def test_basic_functionality():
    """测试基本功能 - 使用 action_mask"""
    batch_size = 3
    seq_len = 8
    
    # 创建测试数据
    r = [[1.0, 2.0], [3.0], [1.0, 2.0, 3.0]]
    eostep_indices = [[3, 5], [4], [2, 4, 6]]
    kl = torch.ones(batch_size, seq_len)
    action_mask = torch.ones(batch_size, seq_len)
    kl_coef = 0.1
    
    # 计算奖励
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        action_mask=action_mask,
        eostep_indices=eostep_indices
    )
    
    # 验证结果
    assert isinstance(reward, torch.Tensor)
    assert reward.shape == (batch_size, seq_len)
    
    # 验证奖励分配
    expected_rewards = torch.zeros(batch_size, seq_len) - kl_coef
    expected_rewards[0, 3] += 1.0
    expected_rewards[0, 5] += 2.0
    expected_rewards[1, 4] += 3.0
    expected_rewards[2, 2] += 1.0
    expected_rewards[2, 4] += 2.0
    expected_rewards[2, 6] += 3.0
    
    assert torch.allclose(reward, expected_rewards)

def test_without_action_mask():
    """测试不使用 action_mask 的情况"""
    r = [[1.0], [2.0]]
    eostep_indices = [[2], [3]]
    kl = [torch.ones(4), torch.ones(5)]
    num_actions = [4, 5]
    kl_coef = 0.1
    
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        num_actions=num_actions,
        action_mask=None,
        eostep_indices=eostep_indices
    )
    
    assert isinstance(reward, list)
    assert len(reward) == 2
    assert reward[0].shape == (4,)
    assert reward[1].shape == (5,)
    
    # 验证奖励分配
    expected_reward0 = torch.full((4,), -kl_coef)
    expected_reward0[2] += 1.0
    expected_reward1 = torch.full((5,), -kl_coef)
    expected_reward1[3] += 2.0
    
    assert torch.allclose(reward[0], expected_reward0)
    assert torch.allclose(reward[1], expected_reward1)

def test_reward_clipping():
    """测试奖励裁剪功能"""
    batch_size = 2
    seq_len = 6
    
    r = [[1.0, 5.0], [-2.0, 4.0]]
    eostep_indices = [[2, 4], [1, 3]]
    kl = torch.ones(batch_size, seq_len)
    action_mask = torch.ones(batch_size, seq_len)
    kl_coef = 0.1
    reward_clip_range = (-1.0, 3.0)
    
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        action_mask=action_mask,
        eostep_indices=eostep_indices,
        reward_clip_range=reward_clip_range
    )
    
    # 验证结果被裁剪
    expected_rewards = torch.zeros(batch_size, seq_len) - kl_coef
    expected_rewards[0, 2] += 1.0
    expected_rewards[0, 4] += 3.0  # 5.0 被裁剪到 3.0
    expected_rewards[1, 1] += -1.0  # -2.0 被裁剪到 -1.0
    expected_rewards[1, 3] += 3.0  # 4.0 被裁剪到 3.0
    
    assert torch.allclose(reward, expected_rewards)

def test_zero_kl_coef():
    """测试 KL 系数为 0 的情况"""
    batch_size = 2
    seq_len = 4
    
    r = [[1.0], [2.0]]
    eostep_indices = [[1], [2]]
    kl = torch.ones(batch_size, seq_len)
    action_mask = torch.ones(batch_size, seq_len)
    kl_coef = 0.0
    
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        action_mask=action_mask,
        eostep_indices=eostep_indices
    )
    
    # 验证结果只包含原始奖励，没有 KL 惩罚
    expected_rewards = torch.zeros(batch_size, seq_len)
    expected_rewards[0, 1] = 1.0
    expected_rewards[1, 2] = 2.0
    
    assert torch.allclose(reward, expected_rewards)

def test_negative_kl_coef():
    """测试负的 KL 系数会被设为 0"""
    batch_size = 2
    seq_len = 4
    
    r = [[1.0], [2.0]]
    eostep_indices = [[1], [2]]
    kl = torch.ones(batch_size, seq_len)
    action_mask = torch.ones(batch_size, seq_len)
    kl_coef = -0.1
    
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        action_mask=action_mask,
        eostep_indices=eostep_indices
    )
    
    # 验证结果与 kl_coef = 0 的情况相同
    expected_rewards = torch.zeros(batch_size, seq_len)
    expected_rewards[0, 1] = 1.0
    expected_rewards[1, 2] = 2.0
    
    assert torch.allclose(reward, expected_rewards)

def test_empty_rewards():
    """测试空奖励序列"""
    batch_size = 2
    seq_len = 4
    
    r = [[], [1.0]]
    eostep_indices = [[], [2]]
    kl = torch.ones(batch_size, seq_len)
    action_mask = torch.ones(batch_size, seq_len)
    kl_coef = 0.1
    
    reward = compute_reward_new(
        r=r,
        kl_coef=kl_coef,
        kl=kl,
        action_mask=action_mask,
        eostep_indices=eostep_indices
    )
    
    # 验证结果
    expected_rewards = torch.full((batch_size, seq_len), -kl_coef)
    expected_rewards[1, 2] += 1.0
    
    assert torch.allclose(reward, expected_rewards)

def test_find_numbers():
    print(find_numbers(".80"))

def test_solution_end_is_valid():
    print(solution_end_is_valid("####.8"))

if __name__ == "__main__":
    test_solution_end_is_valid()