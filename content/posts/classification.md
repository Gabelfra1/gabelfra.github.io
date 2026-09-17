---
author: "Francesco Gabellini"
title: "Classification is back"
date: "2026-09-17"
tags: 
- LLM
- Calibration
- Agent
---

### The decision was always the product

A while ago I wrote about [how badly calibrated LLMs are](/blog/llm-are-way-too-confident). Any GPT-model beat a traditional ML model (TF-IDF & Naive Bayes) at classifcation but it's probability scores are just too uncalibrated to be useful.

Then I wrote a [code along on agentic patterns](/blog/building-effective-agents), and every one of those patterns has the same problem underneath. Routing is a classifier, the evaluator is a classifier, the tool selector is a classifier and so on..

Put the two articles together and you have the real problem with modern automation. What we build is workflows. What a workflow needs at every node is a decision with an honest probability attached. What the model is optimised to produce is text not calibrated decisions.

#### What automation actually asks for

Strip the marketing off any enterprise use case and the loop is the same. Ingest some context, most of it unstructured. Make a decision. Act on it with a tool. Repeat.

None of those steps needs the capability to produce text. Nobody in production reads the model's reasoning trace of why it routed the ticket to the credit card team. The explanation exists because text is the only interface on offer, so we generate it and throw it away after parsing the output to generate a structured response.

The parsing is the tell. If your first action on receiving a model output is to run a validator over it, the model is not speaking your language.

#### The hoops

Everyone building these systems carries the same scaffolding. A Pydantic schema plus some additional tricks , because the model can emit a category that does not exist. Temperature zero, which does not make the system deterministic, it just makes it feel deterministic. 
Retry loops, so you make five calls instead of one.

This is not a prompting problem. [Guo et al. (2017)](https://arxiv.org/abs/1706.04599) showed that modern networks drift towards overconfidence as a property of how we train them, and RLHF pushes the same way by optimising for what a human rater prefers to read. A model rewarded for sounding helpful will sound certain.

#### System One models

TypeSafe released [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) on 15 September, the first of what they call **System One models**. The name comes from Kahneman's split between fast intuitive judgement and slow deliberate reasoning.

The design decision is the interesting part. Jev gives up text generation entirely. It samples all of its outputs in parallel in a single query rather than one token at a time, and it is trained with a method they call Reinforcement Learning for Calibrated Decisions instead of RLHF. You send a state and a set of typed questions, and you get back typed values and probability distributions.

Everything in the previous section becomes unnecessary if that holds. There is no parsing because there is no string. There are no type errors, not because they are rare but because the possible outputs are defined in advance and schema matching is guaranteed. And the probability is not fugazi, it is what the model was trained to produce.

#### Calibration is king

**Calibration means that among all the predictions the model assigns 70% to, roughly 70% turn out correct.** Without that property you cannot set a threshold, and without a threshold every decision needs a human in the loop.

Jav returns the full distribution and also collapses its shape into a single confidence number, so you can threshold without computing it yourself. A flat distribution means low confidence: no option is a clear winner, or the state does not contain enough to go on. What follows is the pattern every risk function already understands.
A floor below which you route to a human, and above it a bar for acting without confirmation that scales with what the action costs if it is wrong. 

#### The part I am not sold on yet

The workflow evals are the weak point, agreement with a panel of frontier LLMs is not the same as being right (try LLM as a judge), and it is a strange yardstick for a model whose pitch is that frontier LLMs are badly calibrated. The one claim genuinely free of this problem is type safety, which is guaranteed by construction and falsifiable with a single API call.

And the obvious one. This does not replace LLMs. It replaces the part of your agent that was pretending to be a classifier. When you need generated language,images or video you still need a generative model. What changes is that the decision layer stops being made of the same material as the generation layer.

What I want to see, and intend to run myself (if I get my hand on the early access), is a straight calibration benchmark. Same 20 Newsgroups setup as the first article, reliability diagram, ECE, Brier score, no vendor in the loop.

### Conclusion

I ended the calibration article by saying that where a decision depends on a trustworthy measure of confidence, modern LLMs are a bad choice, and you should accept the cost of labels and training something simpler. I still think that was right for the world as it was.

But if the generality (of inputs) and the calibration (of outputs) hold up across common domains, RLCD could turn out to be the missing piece in every automation workflow.

And I would be very happy to delete the scaffolding we built around LLMs to force them into being type safe and marginally less overconfident, and to pay less for the privilege.