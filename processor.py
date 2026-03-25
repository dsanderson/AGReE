class Processor:
    def __init__(self, processor_function):
        self.processor_function = processor_function
    
    def __call__(self, inp):
        for item in inp:
            yield self.processor_function(item)
        
    def __or__(self, next):
        return Pipeline(self, next)

class Pipeline(Processor):
    def __init__(self, first, second):
        self.first, self.second  = first, second
    
    def __call__(self, inp):
        return self.second(self.first(inp))
    
class Nest(Processor):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, inp):
        for item in inp:
            list(self.pipeline([item]))
            yield item

class Filter(Processor):
    def __init__(self, filter_func):
        self.filter_func = filter_func

    def __call__(self, inp):
        for item in inp:
            if self.filter_func(item):
                yield item