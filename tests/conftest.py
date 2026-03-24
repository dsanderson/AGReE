import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run LLM tests with real API calls instead of mocks",
    )


@pytest.fixture
def mock_litellm(request, monkeypatch):
    """Mock litellm.completion by default; pass --live to use real APIs."""
    if request.config.getoption("--live"):
        yield
        return

    call_count = [0]

    def mock_completion(messages, model, response_format=None, **kwargs):
        schema = response_format.model_json_schema()
        enum_vals = schema.get("properties", {}).get("label", {}).get("enum", ["default"])
        # enum_vals may be a list of strings or a list containing a tuple (from Literal[tuple(...)])
        flat = []
        for v in enum_vals:
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        # Cycle through labels so rater1/rater2 get different values → produces disagreements
        label = flat[call_count[0] % len(flat)]
        call_count[0] += 1

        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        resp.choices[0].message.content = json.dumps({"label": label})
        return resp

    monkeypatch.setattr("litellm.completion", mock_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.001)
    yield
