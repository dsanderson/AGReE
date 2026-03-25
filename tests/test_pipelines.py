import json
import random
from pathlib import Path

import processor
import persistance
import coder
import review
import utilities

DATA = Path(__file__).parent / "data"


def test_pipeline():
    random.seed(100)

    pl1 = processor.Processor(lambda x: x * 2) | processor.Processor(lambda x: x * 2)
    pl2 = processor.Processor(lambda x: x + 1) | processor.Processor(lambda x: x * 2)
    pl3 = processor.Processor(lambda x: x * 4)

    assert list(pl1([1])) == [4]
    assert list(pl1([2])) == [8]
    assert list(pl1([1, 2])) == [4, 8]
    assert list(pl2([2])) == [6]
    rl = [random.random() for _ in range(10)]
    assert list(pl1(rl)) == list(pl3(rl))

    p1 = processor.Processor(lambda x: x * 2)
    p2 = processor.Processor(lambda x: x + 1)
    pl4 = p1 | p2
    pl5 = p2 | utilities.Progress() | p1
    assert sum(pl4(rl)) + len(rl) == sum(pl5(rl))


def test_jsonl(tmp_path):
    in_file = DATA / "test_num_in.jsonl"
    out_file = tmp_path / "test_num_out.jsonl"

    pl = persistance.JsonlSource(in_file) | persistance.JsonlSink(out_file, reset=True)
    back = list(pl([]))
    assert sum(back) == 15
    res = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert res == back

    pl = persistance.JsonlSource(in_file) | persistance.JsonlSink(out_file)
    back = list(pl([]))
    assert sum(back) == 15
    res = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert sum(res) == sum(back) * 2


def test_coder(mock_litellm, tmp_path):
    instructions = """You will be given a snippet of text from a short story.  Please apply one of the below labels to the snippet, based on the explination for the labels.  Use the specified response format."""
    labels = [
        ("positive", "A good event happens to the characters in the story"),
        ("negative", "A bad event happens to the characters in the story"),
        ("none", "No events happen, or the events are not positive or negative for the characters in the story"),
    ]
    c1 = coder.Coder(instructions, labels, model="openai/gpt-4o-mini")
    c2 = coder.Coder(instructions, labels, model="anthropic/claude-haiku-4-5")

    with open(DATA / "short_story.txt") as f:
        raw = f.read()
    txts = [t.strip() for t in raw.split(".") if t.strip()]

    resp = list(c1([txts[0]]))[0]
    assert resp.get("result")

    resp = list(c2([txts[0]]))[0]
    assert resp.get("result")

    out_file = tmp_path / "test_rater_out.jsonl"
    pl = coder.Rater(c1, c2) | utilities.Progress() | persistance.JsonlSink(str(out_file), reset=True)

    score = review.cohens_kappa(pl(txts[:10]), labels)
    print(f"Cohen's Kappa: {score}")

    dis = review.aggregate_disagreements(persistance.JsonlSource(str(out_file))([]))
    assert len(dis) > 0
