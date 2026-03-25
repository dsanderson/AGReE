from processor import Processor, Filter
import litellm
from typing import Literal
import typing
import time
from pydantic import BaseModel
import json
from dotenv import load_dotenv
load_dotenv()

def labeling_closure(instructions, labels, model, getter=lambda x: x):
    instructions = instructions
    labels = labels
    system_prompt = instructions + "\nLabel - Description\n" + "\n".join(f"{l[0]} - {l[1]}" for l in labels)
    model = model

    class Labels(BaseModel):
        label: Literal[tuple(l[0] for l in labels)]
    
    #print(Labels.model_json_schema())

    def labeling_inner(item):
        messages = [
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':getter(item)}
        ]
        #print(messages)
        response = litellm.completion(
            messages = messages,
            model = model,
            response_format = Labels
        )
        # TODO: extend response with input messages, labels, time, etc.        
        resp_json = {}

        resp_json['usage'] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        resp_json['input'] = messages
        resp_json['response_format'] = Labels.model_json_schema()
        resp_json['result'] = json.loads(response.choices[0].message.content)['label']
        resp_json['model'] = model
        resp_json['instructions'] = instructions
        resp_json['parent'] = item
        try:
            resp_json['cost'] = litellm.completion_cost(completion_response=response)
        except Exception:
            resp_json['cost'] = None
        return resp_json

    return labeling_inner


class Coder(Processor):
    def __init__(self, instructions, labels, model, getter=lambda x: x):
        super().__init__(labeling_closure(instructions, labels, model, getter))
    

def rating_closure(coder1, coder2, eval_func):
    def rating_inner(item):
        resp1 = list(coder1([item]))[0]
        resp2 = list(coder2([item]))[0]
        resp = {'result': eval_func(resp1['result'], resp2['result']), 'parent': item, 'rater1': resp1, 'rater2': resp2}

        return resp
    return rating_inner


def transformer_closure(instructions, model):
    system_prompt = instructions

    class Responses(BaseModel):
        responses: list[str]

    def transformer_inner(item):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': item}
        ]
        response = litellm.completion(messages=messages, model=model, response_format=Responses)
        resp_json = {
            'result': json.loads(response.choices[0].message.content)['responses'],
            'parent': item,
            'model': model,
            'instructions': instructions,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            },
        }
        try:
            resp_json['cost'] = litellm.completion_cost(completion_response=response)
        except Exception:
            resp_json['cost'] = None
        return resp_json

    return transformer_inner


class Transformer(Processor):
    def __init__(self, instructions, model):
        super().__init__(transformer_closure(instructions, model))


class Rater(Processor):
    def __init__(self, coder1, coder2, eval_func=lambda a, b: a==b):
        super().__init__(rating_closure(coder1, coder2, eval_func))


class Disagreement(Filter):
    def __init__(self):
        super().__init__(lambda x: not x.get('result', True))


