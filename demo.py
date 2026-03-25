"""
Demo: iterative instruction improvement with AGReE.

Loads sentences from a short story, labels event sentiment with two LLM coders,
measures inter-rater agreement, uses disagreements to revise the instructions and
label descriptions, then reruns to see if agreement improves.
"""
from coder import Coder, Rater
from persistance import JsonlSource, JsonlSink
from utilities import Progress
from review import (
    cohens_kappa, aggregate_disagreements,
    summarize_disagreements, propose_revision,
    pluck_for_rater_disagreement,
)

STORY   = 'tests/data/short_story.txt'
ROUND1  = 'tests/data/demo_round1.jsonl'
ROUND2  = 'tests/data/demo_round2.jsonl'
MODEL_A = 'openai/gpt-4o-mini'
MODEL_B = 'anthropic/claude-haiku-4-5'

instructions = (
    "You will be given a snippet of text from a short story. "
    "Apply one of the labels below based on what happens in the snippet."
)
labels = [
    ('positive', 'A good event happens to the characters in the story'),
    ('negative', 'A bad event happens to the characters in the story'),
    ('none',     'No events happen, or the events are not clearly positive or negative'),
]

# ── Load data ────────────────────────────────────────────────────────────────

with open(STORY) as f:
    raw = f.read()
sentences = [s.strip() for s in raw.split('.') if s.strip()][:50]
print(f"Loaded {len(sentences)} sentences.\n")

# ── Round 1 ──────────────────────────────────────────────────────────────────

print("=== Round 1 ===")
pipeline = (
    Rater(Coder(instructions, labels, model=MODEL_A),
          Coder(instructions, labels, model=MODEL_B))
    | Progress()
    | JsonlSink(ROUND1, reset=True)
)
kappa1 = cohens_kappa(pipeline(sentences), labels)
print(f"\nRound 1 kappa: {kappa1:.3f}\n")

# ── Revise ───────────────────────────────────────────────────────────────────

print("=== Revising instructions ===")
disagreements = aggregate_disagreements(JsonlSource(ROUND1))
summary       = summarize_disagreements(disagreements, pluck_for_rater_disagreement, instructions, labels)
revision      = propose_revision(summary, model=MODEL_A, labels=labels)

new_instructions = revision['instructions']
new_labels       = [(l['name'], l['description']) for l in revision['labels']]

print("Revised instructions:")
print(f"  {new_instructions}")
print("Revised labels:")
for name, desc in new_labels:
    print(f"  {name}: {desc}")
print()

# ── Round 2 ──────────────────────────────────────────────────────────────────

print("=== Round 2 ===")
pipeline = (
    Rater(Coder(new_instructions, new_labels, model=MODEL_A),
          Coder(new_instructions, new_labels, model=MODEL_B))
    | Progress()
    | JsonlSink(ROUND2, reset=True)
)
kappa2 = cohens_kappa(pipeline(sentences), new_labels)
print(f"\nRound 2 kappa: {kappa2:.3f}")
print(f"Improvement:   {kappa2 - kappa1:+.3f}")
