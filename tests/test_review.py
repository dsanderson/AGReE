import json
from unittest.mock import MagicMock

import review


INSTRUCTIONS = "Label the sentiment of each snippet."
LABELS = [
    ("positive", "A good event happens"),
    ("negative", "A bad event happens"),
    ("none", "No notable event"),
]
DISAGREEMENTS = {
    "negative, positive": ["the cat found a warm fire", "she won the prize"],
    "negative, none":     ["it started to rain"],
}


# --- summarize_disagreements ---

def test_summarize_includes_instructions():
    md = review.summarize_disagreements(DISAGREEMENTS, lambda x: x, INSTRUCTIONS)
    assert INSTRUCTIONS in md

def test_summarize_includes_label_pairs():
    md = review.summarize_disagreements(DISAGREEMENTS, lambda x: x, INSTRUCTIONS)
    assert "negative, positive" in md
    assert "negative, none" in md

def test_summarize_includes_examples():
    md = review.summarize_disagreements(DISAGREEMENTS, lambda x: x, INSTRUCTIONS)
    assert "the cat found a warm fire" in md
    assert "she won the prize" in md
    assert "it started to rain" in md

def test_summarize_with_labels_includes_descriptions():
    md = review.summarize_disagreements(DISAGREEMENTS, lambda x: x, INSTRUCTIONS, LABELS)
    for name, desc in LABELS:
        assert name in md
        assert desc in md

def test_summarize_without_labels_omits_labels_section():
    md = review.summarize_disagreements(DISAGREEMENTS, lambda x: x, INSTRUCTIONS)
    assert "## Labels" not in md

def test_summarize_pluck_applied():
    parents = [{'text': 'extracted text', 'other': 'ignored'}]
    dis = {"negative, positive": parents}
    md = review.summarize_disagreements(dis, lambda x: x['text'], INSTRUCTIONS)
    assert "extracted text" in md
    assert "ignored" not in md


# --- propose_revision ---

def _mock_completion(content):
    def mock(messages, model, response_format=None, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(content)
        return resp
    return mock


def test_propose_revision_with_labels(monkeypatch):
    returned = {
        'instructions': 'revised instructions',
        'labels': [
            {'name': 'positive', 'description': 'revised good'},
            {'name': 'negative', 'description': 'revised bad'},
            {'name': 'none',     'description': 'revised neutral'},
        ],
    }
    monkeypatch.setattr("litellm.completion", _mock_completion(returned))

    result = review.propose_revision("some summary", "openai/gpt-4o-mini", labels=LABELS)

    assert result['instructions'] == 'revised instructions'
    assert len(result['labels']) == 3
    assert result['labels'][0] == {'name': 'positive', 'description': 'revised good'}


def test_propose_revision_without_labels(monkeypatch):
    returned = {'instructions': 'new instructions'}
    monkeypatch.setattr("litellm.completion", _mock_completion(returned))

    result = review.propose_revision("some summary", "openai/gpt-4o-mini")

    assert result['instructions'] == 'new instructions'
    assert 'labels' not in result


# --- pluck_for_transform_review ---

def test_pluck_transform_review_shows_input_and_output():
    item = {
        'result': 'I want faster feedback',
        'parent': {
            'result': ['I want faster feedback', 'I need clearer goals'],
            'parent': 'The review process is too slow and unclear',
        },
    }
    text = review.pluck_for_transform_review(item)
    assert 'The review process is too slow and unclear' in text
    assert 'I want faster feedback' in text

def test_pluck_transform_review_labels_sections():
    item = {
        'result': 'I want X',
        'parent': {'result': ['I want X'], 'parent': 'original'},
    }
    text = review.pluck_for_transform_review(item)
    assert 'Input' in text
    assert 'Output' in text


# --- pluck_for_rater_disagreement ---

def test_pluck_rater_disagreement_string():
    assert review.pluck_for_rater_disagreement('some coded text') == 'some coded text'

def test_pluck_rater_disagreement_dict_with_result():
    item = {'result': 'the text that was coded', 'other': 'metadata'}
    assert review.pluck_for_rater_disagreement(item) == 'the text that was coded'

def test_pluck_rater_disagreement_dict_without_result():
    item = {'text': 'fallback', 'id': 42}
    result = review.pluck_for_rater_disagreement(item)
    assert isinstance(result, str)  # graceful fallback to str(item)


# ---

def test_propose_revision_passes_summary_to_llm(monkeypatch):
    captured = {}
    def mock(messages, model, response_format=None, **kwargs):
        captured['messages'] = messages
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps({'instructions': 'x'})
        return resp
    monkeypatch.setattr("litellm.completion", mock)

    review.propose_revision("MY SUMMARY", "openai/gpt-4o-mini")

    assert any("MY SUMMARY" in m['content'] for m in captured['messages'])
