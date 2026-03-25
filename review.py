import litellm
import json
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


def cohens_kappa(items, labels):
    """Compute Kohen's Kappa for 2-rater agreement on `labels`.  Generally, anything above .8 is considered good"""
    mapping = {l[0]:i for i, l in enumerate(labels)}
    n_k1 = [0 for _ in labels]
    n_k2 = [0 for _ in labels]
    n_o = 0
    n = 0
    for item in items:
        n+=1
        if item['result']:
            n_o+=1
        n_k1[mapping[item['rater1']['result']]] += 1
        n_k2[mapping[item['rater2']['result']]] += 1
    p_o = n_o/n
    p_e = sum(x[0]*x[1] for x in zip(n_k1, n_k2))/(n**2)
    # print(mapping, n_k1, n_k2, p_o, p_e)
    return (p_o-p_e)/(1-p_e)


def aggregate_disagreements(items, n_examples=2, order_independent=True):
    disagreements = {}
    for item in items:
        if not item['result']:
            key = [item['rater1']['result'], item['rater2']['result']]
            if order_independent:
                key = sorted(key)
            key = ', '.join(key)
            if key not in disagreements:
                disagreements[key] = []
            if len(disagreements[key]) < n_examples:
                disagreements[key].append(item['parent'])
    return disagreements


def summarize_disagreements(disagreements, pluck, instructions, labels=None):
    """Build a Markdown summary of rater disagreements for passing to an LLM.

    disagreements: output of aggregate_disagreements
    pluck: function to extract display text from a parent item
    instructions: the coding instructions that produced the disagreements
    labels: optional list of (name, description) tuples
    """
    lines = ["# Disagreement Summary", "", "## Original Instructions", "", instructions]

    if labels:
        lines += ["", "## Labels", ""]
        for name, description in labels:
            lines.append(f"- **{name}**: {description}")

    lines += ["", "## Disagreements", ""]
    for label_pair, examples in disagreements.items():
        lines.append(f"### {label_pair}")
        lines.append("")
        for i, parent in enumerate(examples, 1):
            lines.append(f"**Example {i}:**")
            lines.append("")
            lines.append(str(pluck(parent)))
            lines.append("")

    return "\n".join(lines)


def pluck_for_transform_review(item):
    """Pluck text for disagreements where a Rater evaluated Transformer output.

    Expects an Expand output item: {'result': individual_string, 'parent': transformer_output}
    where transformer_output contains {'parent': original_input, ...}.
    """
    original = item['parent']['parent']
    transformed = item['result']
    return f"**Input:** {original}\n\n**Output:** {transformed}"


def pluck_for_rater_disagreement(item):
    """Pluck text for disagreements where a Rater directly coded items.

    Expects the raw item that was passed to the Rater (= input to each Coder).
    """
    if isinstance(item, dict):
        return item.get('result', str(item))
    return item


class _LabelRevision(BaseModel):
    name: str
    description: str

class _RevisionWithLabels(BaseModel):
    instructions: str
    labels: list[_LabelRevision]

class _RevisionInstructionsOnly(BaseModel):
    instructions: str


def propose_revision(summary, model, labels=None):
    """Pass a disagreement summary to an LLM and return revised instructions/labels.

    summary: Markdown string from summarize_disagreements
    model: litellm-compatible model ID
    labels: if provided, the LLM will also revise label descriptions
    Returns: dict with 'instructions' and, if labels were given, 'labels': [{'name', 'description'}]
    """
    response_format = _RevisionWithLabels if labels else _RevisionInstructionsOnly
    suffix = " and label descriptions" if labels else ""
    system_prompt = (
        f"You are an expert at qualitative coding and inter-rater agreement. "
        f"Review the disagreement summary and propose revised instructions{suffix} "
        f"that would help two independent raters reach better agreement."
    )
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': summary},
    ]
    response = litellm.completion(messages=messages, model=model, response_format=response_format)
    return json.loads(response.choices[0].message.content)
