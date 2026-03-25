import json
from pathlib import Path
from processor import Processor, Pipeline

class JsonlSource(Pipeline):
    def __init__(self, source: Path):
        self.source_file = open(source)

    def __call__(self, _):
        for line in self.source_file:
            yield json.loads(line)

    def __iter__(self):
        return self(None)

class JsonlSink(Pipeline):
    def __init__(self, source: Path, reset = False):
        if reset:
            with open(source, 'w') as f:
                pass
        self.source_file = open(source, 'a', buffering=1) #buffer per-line

    def __call__(self, inp):
        for item in inp:
            self.source_file.write(f"{json.dumps(item)}\n")
            yield item
