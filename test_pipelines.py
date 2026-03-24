import processor
import persistance
import coder
import utilities
import random
from pathlib import Path
import json

def test_pipeline():
    random.seed(100)

    pl1 = processor.Processor(lambda x: x*2) | processor.Processor(lambda x: x*2)
    pl2 = processor.Processor(lambda x: x+1) | processor.Processor(lambda x: x*2)
    pl3 = processor.Processor(lambda x: x*4)

    assert(list(pl1([1])) == [4])
    assert(list(pl1([2])) == [8])
    assert(list(pl1([1, 2])) == [4, 8])
    assert(list(pl2([2])) == [6])
    rl = [random.random() for _ in range(10)]
    assert(list(pl1(rl))==list(pl3(rl)))

    p1 = processor.Processor(lambda x: x*2)
    p2 = processor.Processor(lambda x: x+1)
    pl4 = p1 | p2
    pl5 = p2 | utilities.Progress() | p1
    #print(sum(pl4(rl)), sum(pl5(rl)), rl)
    assert(sum(pl4(rl))+len(rl)==sum(pl5(rl)))


def test_jsonl():
    base_path = Path('tests/data')
    in_file = base_path / 'test_num_in.jsonl'
    out_file = base_path / 'test_num_out.jsonl'
    pl = persistance.JsonlSource(in_file) | persistance.JsonlSink(out_file, reset = True)
    back = list(pl([]))
    assert(sum(back)==15)
    res = []
    with open(out_file) as f:
        for line in f:
            # print(line)
            res.append(json.loads(line))
    # print(res, back)
    assert(res == back)

    pl = persistance.JsonlSource(in_file) | persistance.JsonlSink(out_file)
    back = list(pl([]))
    assert(sum(back)==15)
    res = []
    with open(out_file) as f:
        for line in f:
            res.append(json.loads(line))
    assert(sum(res) == sum(back)*2)


def test_coder():
    instructions = """You will be given a snippet of text from a short story.  Please apply one of the below labels to the snippet, based on the explination for the labels.  Use the specified response format."""
    labels = [
        ('positive', 'A good event happens to the characters in the story'),
        ('negative', 'A bad event happens to the characters in the story'),
        ('none', 'No events happen, or the events are not positive or negative for the characters in the story')
    ]
    c1 = coder.Coder(instructions, labels, model="openai/gpt-5-mini")
    c2 = coder.Coder(instructions, labels, model="anthropic/claude-haiku-4-5")
    with open('tests/data/short_story.txt') as f:
        raw = f.read()
    txts = [t.strip() for t in raw.split('.')]
    print(len(txts), txts[0])

    resp = list(c1([txts[0]]))[0]
    print(resp)
    assert(resp.get('label'))

    resp = list(c2([txts[0]]))[0]
    print(resp)
    assert(resp.get('label'))

    pl = coder.Rater(c1, c2) | utilities.Progress() | persistance.JsonlSink('tests/data/test_rater_out.jsonl', reset=True)

    score = coder.cohens_kappa(pl(txts[:50]), labels)
    print(score)

    dis = coder.aggregate_disagreements(persistance.JsonlSource('tests/data/test_rater_out.jsonl')([]))
    assert(len(dis)>0)
    print(dis)



if __name__=="__main__":
    test_pipeline()
    test_jsonl()
    test_coder()