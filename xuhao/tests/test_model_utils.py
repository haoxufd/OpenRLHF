import torch
from openrlhf.models.utils import compute_reward_new
import pytest

def test_compute_reward_new():
    r = [[1.0, 1.0, 1.0], [1.0, 1.0]]
    eostep_indices = [[1, 3, 5], [1, 3]]
    kl_coef = 0.1
    kl = torch.tensor([[0.1, 0.2, 0, 0, 0, 0, 0], [0.3, 0.4, 0, 0, 0, 0, 0]])
    action_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
    reward_clip_range = (0.0, 5.0)
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range=reward_clip_range)
    expected_reward = torch.tensor([[-0.01, -0.02 + 1, 0, 1, 0, 0, 1], [-0.03, -0.04 + 1, 0, 0, 1, 0, 0]])
    assert torch.allclose(reward, expected_reward), f"Expected {expected_reward}, but got {reward}"

def test_minimal_input():
    # 最小输入测试
    r = [[3.0]]
    eostep_indices = [[0]]  # 会被 action_mask 计算得到的 eos index 覆盖
    kl_coef = 0.2
    kl = torch.tensor([[0.5, 0.5, 0.5]])
    action_mask = torch.tensor([[1, 1, 1]])  # eos index = 2
    reward_clip_range = (0.0, 10.0)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    # 计算过程：
    # kl_reward = -0.2 * [0.5, 0.5, 0.5] = [-0.1, -0.1, -0.1]
    # eostep_indices 变为 [2]，将 r[0]=3.0 分配到索引 2 上，则该位置累加 3.0
    # 最终 reward = [ -0.1, -0.1, 3.0-0.1 ] = [ -0.1, -0.1, 2.9 ]
    expected = torch.tensor([[-0.1, -0.1, 2.9]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_reward_clipping():
    # 测试奖励剪裁：奖励超出 clip 范围后被限制在 [0,5] 内
    r = [[-1.0, 10.0]]
    eostep_indices = [[0, 2]]  # 最后一个索引将被替换为 eos index
    kl_coef = 0.0
    kl = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    action_mask = torch.tensor([[1, 1, 1, 1]])  # eos index = 3
    reward_clip_range = (0.0, 5.0)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    # 奖励经过剪裁后：-1.0->0.0, 10.0->5.0；索引变为 [0, 3]，
    # 最终 reward 在 batch 内：[0.0, 0.0, 0.0, 5.0]
    expected = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_multiple_batches():
    # 测试多个 batch 及不同 action_mask 情况
    r = [
        [2.0, 3.0],  # Batch 0
        [4.0]        # Batch 1
    ]
    eostep_indices = [
        [1, 2],      # Batch 0: 第二个索引会被替换为 eos index
        [2]          # Batch 1: 索引会被替换为 eos index
    ]
    kl_coef = 0.5
    kl = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],  # Batch 0
        [0.1, 0.1, 0.1, 0.1]   # Batch 1
    ])
    # 对应 action_mask：
    # Batch 0: [1,1,1,1] -> eos index = 3
    # Batch 1: [1,1,1,0] -> 翻转后 [0,1,1,1]，argmax 返回索引 1，故 eos index = 4-1-1 = 2
    action_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ])
    reward_clip_range = (0.0, 10.0)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    # Batch 0:
    # eostep_indices 变为 [1, 3]，r[0]=[2.0,3.0]分配到索引 1 与 3
    # kl_reward = -0.5 * [0.2,0.2,0.2,0.2] = [-0.1, -0.1, -0.1, -0.1]
    # 最终 row0 = [ -0.1, 2.0-0.1, -0.1, 3.0-0.1 ] = [ -0.1, 1.9, -0.1, 2.9 ]
    # Batch 1:
    # eostep_indices 变为 [2]，r[1]=[4.0] 分配到索引 2
    # kl_reward = -0.5 * [0.1,0.1,0.1,0.1] = [-0.05, -0.05, -0.05, -0.05]
    # 最终 row1 = [ -0.05, -0.05, 4.0-0.05, -0.05 ] = [ -0.05, -0.05, 3.95, -0.05 ]
    expected = torch.tensor([
        [-0.1, 1.9, -0.1, 2.9],
        [-0.05, -0.05, 3.95, -0.05]
    ])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_negative_kl_coef():
    # 测试负的 kl_coef（应置为 0），使 kl_reward 为 0
    r = [[1.0, 2.0]]
    eostep_indices = [[0, 2]]  # 第二个索引被替换为 eos index，依然为 2
    kl_coef = -0.1  # 负值应置为 0
    kl = torch.tensor([[0.5, 0.5, 0.5]])
    action_mask = torch.tensor([[1, 1, 1]])  # eos index = 2
    reward_clip_range = (0.0, 10.0)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    # 此时 kl_reward = 0，r[0] = [1.0, 2.0] 分别分配到索引 0 与 2
    expected = torch.tensor([[1.0, 0.0, 2.0]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_mismatched_lengths():
    # 测试当 r 与 eostep_indices 长度不匹配时是否抛出错误
    r = [[1.0, 2.0, 3.0]]
    eostep_indices = [[1, 2]]  # 长度不匹配（r[0] 有 3 个奖励，但索引只有 2 个）
    kl_coef = 0.1
    kl = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    action_mask = torch.tensor([[1, 1, 1, 1]])
    reward_clip_range = (0.0, 10.0)
    
    with pytest.raises(RuntimeError):
        # 由于形状不匹配，index_put 操作应抛出错误
        _ = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)

def test_accumulation():
    # 测试同一位置奖励累加
    # 构造一个 action_mask，使得 eos index 为 0，从而 eostep_indices 被设置为 [0, 0]
    r = [[1.0, 2.0]]
    eostep_indices = [[0, 0]]
    # 构造 action_mask，使得：action_mask = [1, 0, 0]
    # 计算过程：fliplr([1,0,0]) = [0,0,1]，argmax 返回 2，故 eos index = 3 - 1 - 2 = 0
    action_mask = torch.tensor([[1, 0, 0]])
    kl_coef = 0.1
    kl = torch.tensor([[0.0, 0.0, 0.0]])
    reward_clip_range = (0.0, 10.0)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    # 结果：两个奖励均分配到索引 0，因此该位置奖励应累加 1.0+2.0=3.0，加上 kl_reward 0
    expected = torch.tensor([[3.0, 0.0, 0.0]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_fixed_clip_range():
    # 测试当 reward_clip_range 的最小值和最大值相同时，所有奖励均被固定为该值
    r = [[3.0, 4.0]]
    eostep_indices = [[0, 1]]
    kl_coef = 0.0
    kl = torch.tensor([[0.0, 0.0]])
    action_mask = torch.tensor([[1, 1]])
    reward_clip_range = (2.0, 2.0)  # 无论输入是多少，剪裁后均为 2.0
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    expected = torch.tensor([[2.0, 2.0]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_integer_rewards():
    # 测试整数类型奖励也能正确转换为 float
    r = [[1, 2]]
    eostep_indices = [[0, 1]]
    kl_coef = 0.0
    kl = torch.tensor([[0, 0]])
    action_mask = torch.tensor([[1, 1]])
    reward_clip_range = (0, 10)
    
    reward = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(reward, expected), f"Expected {expected}, but got {reward}"

def test_action_mask_none():
    # 测试当 action_mask 为 None 时，函数应因调用 size() 等操作而抛出 AttributeError
    r = [[1.0]]
    eostep_indices = [[0]]
    kl_coef = 0.1
    kl = torch.tensor([[0.1]])
    action_mask = None
    reward_clip_range = (0.0, 10.0)
    
    with pytest.raises(AttributeError):
        _ = compute_reward_new(r, eostep_indices, kl_coef, kl, action_mask, reward_clip_range)

if __name__ == "__main__":
    pytest.main([__file__])
