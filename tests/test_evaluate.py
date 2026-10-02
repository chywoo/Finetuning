import pytest

from finetune_lab.evaluate import normalize_answer, score_answers, token_f1


def test_answer_normalization():
    assert normalize_answer(" The, CAT! ") == "cat"
    assert normalize_answer("  두 개  ") == "두 개"


def test_token_f1_multiset_and_empty():
    assert token_f1("red red blue", "red blue") == pytest.approx(0.8)
    assert token_f1("", "") == 1.0
    assert token_f1("a dog", "cat") == 0.0


def test_score_answers_counts_and_rejects_mismatch():
    result = score_answers(["The Cat", "blue"], ["cat", "red"])
    assert result["exact_match"] == 0.5
    assert result["token_f1"] == 0.5
    with pytest.raises(ValueError):
        score_answers([], [])
    with pytest.raises(ValueError):
        score_answers(["cat"], [])
