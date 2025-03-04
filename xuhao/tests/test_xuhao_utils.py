from xuhao.utils import get_eostep_indices
import pytest


def test_get_eostep_indices_empty_sequences_returns_empty_list():
    response_sequences = []
    step_split_token_id = 10
    result = get_eostep_indices(response_sequences, step_split_token_id)
    assert result == []


def test_get_eostep_indices_no_split_token_returns_empty_sublists():
    response_sequences = [[1, 2, 3], [4, 5, 6]]
    step_split_token_id = 10
    result = get_eostep_indices(response_sequences, step_split_token_id)
    assert result == [[], []]


def test_get_eostep_indices_single_split_token_returns_single_index():
    response_sequences = [[1, 2, 10, 3], [4, 10, 5, 6]]
    step_split_token_id = 10
    result = get_eostep_indices(response_sequences, step_split_token_id)
    assert result == [[2], [1]]


def test_get_eostep_indices_multiple_split_tokens_returns_multiple_indices():
    response_sequences = [[1, 10, 2, 10, 3], [10, 4, 10, 5, 10]]
    step_split_token_id = 10
    result = get_eostep_indices(response_sequences, step_split_token_id)
    assert result == [[1, 3], [0, 2, 4]]


def test_get_eostep_indices_mixed_sequences_returns_correct_indices():
    response_sequences = [[1, 2, 3], [10, 4, 10, 5], [6, 7, 8]]
    step_split_token_id = 10
    result = get_eostep_indices(response_sequences, step_split_token_id)
    assert result == [[], [0, 2], []]