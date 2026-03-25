from processor import Processor
import time
import math
import traceback

class Catch(Processor):
    def __init__(self, pipeline, errors=None):
        self.pipeline = pipeline
        self.errors = errors

    def __call__(self, inp):
        for item in inp:
            try:
                yield from self.pipeline([item])
            except Exception as e:
                if self.errors:
                    list(self.errors([{
                        'item': item,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }]))


def _sum_costs(item):
    total = 0
    while isinstance(item, dict):
        cost = item.get('cost')
        if cost is not None:
            total += cost
        item = item.get('parent')
    return total


class Progress(Processor):
    def __init__(self, refresh_s = 0.5):
        self.refresh_s = refresh_s

    def __call__(self, items):
        start = time.time()
        count = 0
        total_cost = 0
        last = math.inf
        for item in items:
            count += 1
            total_cost += _sum_costs(item)
            if last-start > self.refresh_s:
                last = time.time()
                rate = count/(last-start)
                cost_str = f", ${total_cost:.4f}" if total_cost else ""
                print(f"\rProcessed {count}, Running for {last-start:0.1f}s, {rate:0.2f}/s{cost_str}", end="")
            yield item