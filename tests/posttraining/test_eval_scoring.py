"""Format scorer must not conflate 'weak stop' with 'wrong answer'."""
from src.posttraining.eval import _score_case


def test_correct_answer_with_trailing_template_does_not_zero_score():
    case = {
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "expected_format": "short_answer",
        "gold": "Paris",
        "max_tokens": 6,
    }
    score = _score_case(case, response="Paris", raw_generated="Paris\n\n### Instruction:")
    assert score == 1.0


def test_wrong_answer_with_trailing_template_scores_zero():
    case = {
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "expected_format": "short_answer",
        "gold": "Paris",
        "max_tokens": 6,
    }
    score = _score_case(case, response="Berlin", raw_generated="Berlin\n\n### Instruction:")
    assert score == 0.0


def test_response_with_inline_forbidden_substring_still_zeros():
    case = {
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "expected_format": "short_answer",
        "gold": "Paris",
        "max_tokens": 6,
    }
    score = _score_case(case, response="### Instruction: Paris", raw_generated="### Instruction: Paris")
    assert score == 0.0
