from xuhao.utils import get_eostep_indices, is_number, solution_is_valid


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

def test_is_number_integer():
    assert is_number("123") == True
    assert is_number("123456") == True
    assert is_number("123,456") == True
    assert is_number("12,3456") == False

def test_is_number_float():
    assert is_number("123.456") == True

def test_is_number_comma_separated_number():
    assert is_number("1,234.56") == True
    assert is_number("12,34.56") == False

def test_is_number_fraction():
    assert is_number("1/2") == True
    assert is_number("10/11") == True
    assert is_number("1/0") == True

def test_is_number_non_numeric_string():
    assert is_number("abc") == False

def test_is_number_empty_string():
    assert is_number("") == False

def test_is_number_whitespace_string():
    assert is_number("   ") == False

def test_solution_is_valid():
    # valid solution
    solution_1 = "First find the total number of calories in the carrots: 20 calories/carrot * 5 carrots = <<20*5=100>>100 calories\n<|reserved_special_token_0|>Then subtract the calories in the burger and carrots from the total number of calories to find the number of calories in the cookies: 750 calories - 400 calories - 100 calories = <<750-400-100=250>>250 calories\n<|reserved_special_token_0|>Then divide the total number of calories by the number of calories per cookie to find the number of cookies each kid gets: 250 calories / 50 calories/cookie = <<250/50=5>>5 cookies\n#### 5<|reserved_special_token_0|><|eot_id|>"
    # invalid solution with wrong ending <|reserved_special_token_0|><|reserved_special_token_0|>
    solution_2 = "First find the total number of calories in the carrots: 20 calories/carrot * 5 carrots = <<20*5=100>>100 calories\n<|reserved_special_token_0|>Then subtract the calories in the burger and carrots from the total number of calories to find the number of calories in the cookies: 750 calories - 400 calories - 100 calories = <<750-400-100=250>>250 calories\n<|reserved_special_token_0|>Then divide the total number of calories by the number of calories per cookie to find the number of cookies each kid gets: 250 calories / 50 calories/cookie = <<250/50=5>>5 cookies\n#### 5<|reserved_special_token_0|><|reserved_special_token_0|>"
    # invalid solution with extra <|reserved_special_token_0|> after the last step
    solution_3 = "First find the total number of calories in the carrots: 20 calories/carrot * 5 carrots = <<20*5=100>>100 calories\n<|reserved_special_token_0|>Then subtract the calories in the burger and carrots from the total number of calories to find the number of calories in the cookies: 750 calories - 400 calories - 100 calories = <<750-400-100=250>>250 calories\n<|reserved_special_token_0|>Then divide the total number of calories by the number of calories per cookie to find the number of cookies each kid gets: 250 calories / 50 calories/cookie = <<250/50=5>>5 cookies\n<|reserved_special_token_0|>#### 5<|reserved_special_token_0|><|eot_id|>"
    # invalid solution with repetition content
    solution_4 = "Layla earned $105 from the Donaldsons because 7 �~W 15 = <<7*15=105>>105\n<|reserved_special_token_0|>Layla earned $108 from the Merck family because 6 �~W 18 = <<6*18=108>>108\n<|reserved_special_token_0|>Layla earned $60 from the Hille family because 3 �~W 20 = <<3*20=60>>60\n<|reserved_special_token_0|>Layla earned $273 in total because 105 + 108 + 60 = <<105+108+60=273>>273\n<|reserved_special_token_0|>Layla earned $168 more babysitting the Merck family than the Donaldsons because 108 - 105 = <<108-105=3>>3\n<|reserved_special_token_0|>Layla earned $48 more babysitting the Hille family than the Merck family because 60 - 108 = <<60-108=-48>>-48\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-45>>-45\n<|reserved_special_token_0|>Layla earned $216 more babysitting the Hille family than the Donaldsons because 60 - 105 = <<60-105=-"

    assert solution_is_valid(solution_1) == True
    assert solution_is_valid(solution_2) == False
    assert solution_is_valid(solution_3) == False
    assert solution_is_valid(solution_4) == False
