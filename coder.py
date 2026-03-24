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
        resp_json['label'] = json.loads(response.choices[0].message.content)['label']
        resp_json['model'] = model
        resp_json['instructions'] = instructions
        resp_json['item'] = item
        try:
            resp_json['cost'] = litellm.completion_cost(completion_response=response)
        except Exception:
            resp_json['cost'] = None
        return resp_json

    return labeling_inner


class Coder(Processor):
    def __init__(self, instructions, labels, model):
        super().__init__(labeling_closure(instructions, labels, model))
    

def rating_closure(coder1, coder2, eval_func):
    def rating_inner(item):
        resp1 = list(coder1([item]))[0]
        resp2 = list(coder2([item]))[0]
        resp = {'rater1':resp1, 'rater2':resp2, 'agreement': eval_func(resp1['label'], resp2['label'])}

        return resp
    return rating_inner


class Rater(Processor):
    def __init__(self, coder1, coder2, eval_func=lambda a, b: a==b):
        super().__init__(rating_closure(coder1, coder2, eval_func))


class Disagreement(Filter):
    def __init__(self):
        super().__init__(lambda x: not x.get('agreement', True))


def cohens_kappa(items, labels):
    """Compute Kohen's Kappa for 2-rater agreement on `labels`.  Generally, anything above .8 is considered good"""
    mapping = {l[0]:i for i, l in enumerate(labels)}
    n_k1 = [0 for _ in labels]
    n_k2 = [0 for _ in labels]
    n_o = 0
    n = 0
    for item in items:
        n+=1
        if item['agreement']:
            n_o+=1
        n_k1[mapping[item['rater1']['label']]] += 1
        n_k2[mapping[item['rater2']['label']]] += 1
    p_o = n_o/n
    p_e = sum(x[0]*x[1] for x in zip(n_k1, n_k2))/(n**2)
    print(mapping, n_k1, n_k2, p_o, p_e)
    return (p_o-p_e)/(1-p_e)

def aggregate_disagreements(items, n_examples = 2, order_independent = True):
    disagreements = {}
    for item in items:
        if not item['agreement']:
            key = [item['rater1']['label'], item['rater2']['label']]
            if order_independent:
                key = sorted(key)
            key = ', '.join(key)
            if key not in disagreements:
                disagreements[key] = []
            if len(disagreements[key])<n_examples:
                disagreements[key].append(item['rater1']['input'])
    return disagreements
