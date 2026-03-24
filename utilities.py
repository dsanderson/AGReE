from processor import Processor
import time
import math

class Progress(Processor):
    def __init__(self, refresh_s = 0.5):
        self.refresh_s = refresh_s

    def __call__(self, items):
        start = time.time()
        count = 0
        last = math.inf
        for item in items:
            count += 1
            if last-start > self.refresh_s:
                last = time.time()
                rate = count/(last-start)
                print(f"\rProcessed {count}, Running for {last-start:0.1f}s, {rate:0.2f}/s", end="")
            yield item