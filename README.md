# Automated Grounded Research Environment (AGReE)

LLMs have opened the door of natural language processing using natural language instructions.  This has myriad uses, but one that has proven incredibly powerful is *qualitative coding for grounded research*.  Imagine you're sitting on a pile of qualitative data: user interviews, product reviews, videos of activities, etc.  How do you extract quantitative insights from that data?  You write up instruction for *applying labels* to chunks from the qualitative data, then you can analyze those label quantitatively (thinks like frequency, co-occurance, etc.).  But, are those labels actually meaningful?  The typical way of testing that is to have two different people apply those label (*code* the data), then check the inter-rater agreement with a metric like Cohen's Kappa.  If the kappa is low (<0.8), dig into the disagreements and revise your instructions.  If the kappa is good (>0.8), you're golden.

This project is my attempt to simplify these workflows, using LLMs.  There's a simple format for defining pipelines of data sources, filters, coders, raters and loggers, and utilities for scoring and examining disagrements.

The vision is to make it "easy" to gain insights from huge quantities of qualitative data, by making it easy to develop, test, optimize and run labelling pipelines!

## Example

Given a JSON lines file of snippets from a story, label the events in the snippet as positive, negative or none, and check the Cohen's Kappa for the two.

```python
# Create two coders for event sentiment in a short story
instructions = """You will be given a snippet of text from a short story.  Please apply one of the below labels to the snippet, based on the explination for the labels.  Use the specified response format."""
labels = [
    ('positive', 'A good event happens to the characters in the story'),
    ('negative', 'A bad event happens to the characters in the story'),
    ('none', 'No events happen, or the events are not positive or negative for the characters in the story')
]
openai_coder = Coder(instructions, labels, model="openai/gpt-5-mini")
anthropic_coder = Coder(instructions, labels, model="anthropic/claude-haiku-4-5")

# Create a pipeline to load text from a JSON lines file, perform IRA, save results, printe progress, and compute the score at the end
pipeline =  Rater(openai_coder, anthropic_coder) | Progress() | JsonlSink('output.jsonl')

score = cohens_kappa(pipeline(JsonlSource('input.jsonl')([])), labels)
print(score)
disagreements = aggregate_disagreements(JsonlSource('output.jsonl'))
```

In practice, this gives a pretty bad kappa, like 0.5.  This makes sense, as the descritions are pretty ambiguous; how much is the rater to read between the lines?  That's where the `aggregate_disagreements` function comes in.  It lups together examples from any disagreed label pairs.  You could use this to manually update the instruction, **or** you can pass that info into an LLM as well!  We're working on a prompt template/call template to automate this loop, but either way this provides a simple flow for automated grounded research:

```mermaid
flowchart AGReE;
    A(["Write initial instructions"])
    A-->B["Sample data"]
    B-->G["Run rater on sample data"]
    G-->C{"Check agreement >0.8"}
    C-->D["Collect disagreements"]
    D-->E["Use LLM to update labels/instructions"]
    E-->G
    C-->Z["Run coder on all data"]
```

## Structure
### processor.py

Core structure of the pipeline.  We have two classes:
- `Processor`: takes a per-item processing function, when called with an iterable will return a generator applying that function to each item
- `Pipeline`: a processor that abuses the `__or__` operator to allow pipe-chaining
- `Filter`: a processor that takes a per-item filter function; skips the item if the filter is false-y

## persistance.py

Pipeline elements for loading/saving data.
- `JsonlSource`: loads a JSON lines file; when called, ignores the argument and iterates over each line in the JSONL file.
- `JsonlSink`: takes a JSON lines file; saves each item to a new line in the file, and returns the item.  reset flag will reset the file at initialization.

## utilities.py

Other pipeline utilities.
- `Progress`: show progress through the pipeline in the console

## coder.py

The bulk of the logic for coding and analysis
- `Coder`: given instructions and labels (list of `(label name, label description)`) pairs and the litellm-compatible model ID, create a Processor that will label each item passed in.  We return as much as the input as we can (for repeatability), usage and cost.  The label is in the ['label'] field of the input, and the ['item'] contains the original item passed for coding.  This is mostly done to enable easy filtering + chained coding
- `Rater`: (probably will get renamed) Takes two coders, and an optional evaluation function (eg. if we're doing ordinal coding and want some "dead zone"), and runs each item through both coders, logging if they agree
- `Disagreement`: Basic filter for ratings that disagreed in the pipeline
- `cohens_kappa`: function that tasks a bunch of ratings and the labels and computes the cohen's kappa; generally, >0.8 means we did well!
- `aggregate_disagreements`: function that will aggregate the label pairs and `n` examples of each type of disagreement, so we can understand the errors and update the label descriptions/instructions.  This can be automated!!!

## To Do
- Prompt template for updating instructions autonomously
- Better filter/data plucking utilities
- "tee" utility to allow multiple analysis
  - Could turn functions into pipeline elements, and check the class property later?
  - Could create "nested" pipelines, only get called if parent filter matches, otherwise just return the item?  This seems best!
- Cost mgmt?
- Clean up jsonlsource call
- Caching/resuming
- error handling!!