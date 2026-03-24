# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AGReE** (Automated Grounded Research Environment) — an LLM-powered framework for qualitative data coding and inter-rater agreement analysis. The core workflow: define labels, run two LLM coders against the same data, measure inter-rater agreement via Cohen's Kappa, then iteratively refine instructions until kappa > 0.8.

## Commands

**Package manager:** `uv` (Python 3.12)

```bash
# Install dependencies
uv sync

# Run tests (no test runner configured — tests are called directly)
python test_pipelines.py

# Run a specific test function
python -c "import test_pipelines; test_pipelines.test_pipeline()"
```

**Environment:** Copy `.env` and populate with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (and optionally `TAVILY_API_KEY`). `dotenv` loads these automatically via `load_dotenv()` in `coder.py`.

## Architecture

### Pipeline Composition (`processor.py`)

Everything is a `Processor` — a callable that takes an iterable and returns a generator. Processors chain via the `|` operator (overloaded `__or__`), which creates a `Pipeline` that feeds one processor's output into the next's input. `Filter` is a `Processor` that conditionally yields items.

```python
# Pipelines are lazy generators; nothing executes until consumed
pipeline = SourceA() | TransformB() | SinkC()
results = list(pipeline([]))  # triggers execution
```

### Data Flow for IRA (Inter-Rater Agreement)

1. `JsonlSource` → produces raw items from a `.jsonl` file
2. `Rater(coder1, coder2)` → runs each item through both coders, returns `{rater1, rater2, agreement}` dicts
3. `Progress()` → passes items through while printing throughput stats
4. `JsonlSink` → writes each item to `.jsonl` and passes it through
5. `cohens_kappa(pipeline_output, labels)` → consumes the generator and returns the score
6. `aggregate_disagreements(JsonlSource(...))` → reads saved output and groups disagreement examples by label pair

### `Coder` (`coder.py`)

Uses `litellm` for unified LLM access (supports `openai/...` and `anthropic/...` model IDs). Responses use Pydantic structured output to enforce label constraints. Each coded item returns: `label`, `item` (original), `usage`, `cost`, `input` (messages), `model`, `instructions`.

### Key Design Detail: Sinks consume and re-yield

`JsonlSink` writes items to disk **and returns them**, so pipelines can continue after a sink. This enables saving intermediate results while still computing aggregates:

```python
score = cohens_kappa(
    pipeline(source),  # pipeline includes a sink mid-stream
    labels
)
```
