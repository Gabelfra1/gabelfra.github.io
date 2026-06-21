---
author: "Francesco Gabellini"
title: "Who Watches the Watchmen?"
date: "2026-06-21"
tags: 
- LLM
- Evaluation
---

*This article is about single-turn LLM generation: producing text, answering questions, summarising, and so on. I am not going to cover multi-turn conversation or agentic systems. The question is a practical one. You have shipped, or are about to ship, a system that relies on an LLM for real users in a real business. How do you know if it is any good?*

### The old world: NLP measures

Before large language models entered the picture, NLP evaluation had a clean, mathematical answer: measure how similar the output is to a known good reference. The dominant metric was BLEU ([Papineni et al., 2002](https://aclanthology.org/P02-1040/)), the Bilingual Evaluation Understudy, imported from the machine translation research that would later give us the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). BLEU counts overlapping n-grams between a generated string and a reference string and produces a score between 0 and 1. ROUGE ([Lin, 2004](https://aclanthology.org/W04-1013/)) did the same for summarisation. Cosine similarity over TF-IDF or word embeddings offered a softer version of the same idea.

These metrics had real virtues. They were fast, cheap, deterministic, and needed no human in the loop once you had a reference corpus.

The problem is that language is not a bag of words, and meaning is not proximity in token space. "The patient is not responding to treatment" and "The patient is responding well to treatment" share most of their surface and would score highly against one another. "The medication was administered" and "The medication was not administered" sit a single negation apart. Similarity-based metrics are structurally blind to these distinctions, and in a real world scenario that blindness becomes fatal.

### Closing the gap: Natural Language Inference

The field's next attempt to do better came through Natural Language Inference (NLI): the task of deciding whether a hypothesis is entailed by, contradicted by, or neutral with respect to a premise. Models fine-tuned on datasets like SNLI ([Bowman et al., 2015](https://arxiv.org/abs/1508.05326)) and MNLI ([Williams et al., 2018](https://arxiv.org/abs/1704.05426)) learned something closer to semantic reasoning than to surface matching.

The training data looks like this:

| Premise | Hypothesis | Label |
|---|---|---|
| A football game with kids playing. | Some kids are playing. | Entailment |
| A man is in a kitchen cooking | The man is sleeping. | Contradiction |

The idea, applied to evaluation, was to check not whether the output looked like the reference, but whether it was logically coherent with, it was a genuine improvement. NLI-based scoring caught contradictions that BLEU missed; it was sensitive to negation, to entailment direction, to whether a claim was supported or merely adjacent to the reference.

But NLI brought its own problem, and you can already see it in the table above. Those premises and hypotheses are short, tidy, single-clause sentences derived from image captions. Real documents do not read like the Elements of Euclid. Real world documents are messy, contextual,  sentences long, full of hedges and asides and clauses that depend on one another. An entailment model trained on caption pairs has very little to say about whether a four-paragraph answer faithfully reflects a retrieved policy document. It was a well-designed experiment that, for the most part, stayed in the lab.

### The new world : LLM-as-Judge

Once capable LLMs existed, the obvious move was to use them to evaluate each other. Feed the system prompt, the user query and the model's response to a capable judge model and get back a score. This approach, LLM-as-Judge, was studied systematically by [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685) and has since become the de facto standard. A capable judge can assess fluency, faithfulness, relevance and tone in a single pass, and it can explain its reasoning on edge cases that would confuse any fixed metric.

A growing ecosystem of packages wraps this pattern into something you can drop into a pipeline. RAGAS ([Es et al., 2024](https://arxiv.org/abs/2309.15217)) is one example among many: built for retrieval-augmented generation, it scores dimensions like faithfulness (does the answer stick to the retrieved context?), answer relevancy and context recall, each judged independently so you get a full picture rather than one opaque number. DeepEval, TruLens and others occupy the same space with different trade-offs. **The framework matters less than understanding what it is doing underneath, which is asking an LLM to grade another LLM.**

That is where the real problem lives, and it is one of epistemology: you are using a model with its own biases to evaluate a model with its own biases. Three failure modes are well documented. **Self-preference bias**: judges favour outputs that resemble their own, and [Panickssery et al. (2024)](https://arxiv.org/abs/2404.13076) link this directly to a model's ability to recognise its own generations, so using GPT-4 to judge GPT-4 is not a neutral act. **Verbosity bias**: longer, more hedged answers score higher than concise, correct ones.**Position bias**: in a pairwise comparison the answer shown first wins more often.

Mitigations exist: prompt the judge to reason before scoring, run each comparison in both orders and average, use a different model family as judge than the one under test. None of them fully closes the gap. LLM-as-Judge is powerful tool but has it's cracks.

### Back to basics: the golden dataset

The canonical machine learning answer is the test dataset: a golden dataset of inputs paired with verified outputs, against which any version of the system can be scored and compared. The appeal is obvious. It is deterministic, reproducible and immune to judge bias.

The problem is circular in an uncomfortable way. To score an LLM's free-form generation against a golden answer, you need a method to compare the produced response to the expected one, and you are right back where you started. A similarity metric misses semantic divergence; an NLI model carries the problems described above; a human reviewing every pair does not scale. You can use an LLM judge for the comparison, but then your golden evaluation is only as good as that judge, and you have not escaped the watchmen problem, you have just moved it up one layer.

Golden datasets still earn their place when you narrow what they test. For constrained tasks (extracting a specific field, classifying into a fixed set of categories, requiring a response to contain certain factual claims) binary or near-binary correctness checks are both tractable and meaningful. The failure mode is stretching this to open-ended generation, where the space of acceptable answers is large and your reference is only one valid output among many.

When you do need to measure closeness to an expected answer, the most reliable options today are embedding-based similarity using a strong general-purpose embedding model, or structured decomposition: break the expected answer into individual factual claims and verify each one separately rather than comparing the whole.

### The user as ground truth

There is one more signal, and it sits in production: the user. Thumbs up and thumbs down, copy-to-clipboard events, a follow-up question that reveals the first answer was wrong, session usage. These are all expressions of satisfaction or dissatisfaction that need no instrumentation beyond basic analytics.

The seductive thing about implicit feedback is that it is free, continuous and available at scale. The dangerous thing is that it is confounded by almost everything: UI, task difficulty, user expertise, stylistic preferences that have nothing to do with quality. A thumbs down might mean "the answer was wrong", or "the answer was right but too boring", or "I clicked it by accident", or "I was already frustrated before I opened the tool". Explicit Likert ratings remove some of that ambiguity but add survey fatigue and selection bias, since the users who rate are rarely representative of the users who do not.

Treat user feedback as a monitoring signal, not a validation metric. It is good for catching regressions, surfacing categories of failure and deciding what to investigate next. It is not reliable enough to be the primary basis for an iteration decision.

### A practical framework

Putting it all together, here is a reasonable approach for a team shipping a production LLM system, at least in my opinion

**Define what correctness means before you build anything.** The evaluation strategy should fall out of the task definition, not get bolted on afterwards. If you cannot write down in concrete terms what a good response looks like, you are not ready.

**Use automated metrics as filters, not as truth.** LLM-as-Judge metrics catch obvious failures reliably and subtle ones unreliably. Use them to shrink the volume of output that needs human eyes, not to replace those eyes.

**Build a golden dataset for regression testing** A modest, carefully curated set of inputs with verified references beats a large noisy one.That set is your foundation for changing the system without breaking it in production.

**Show the retrieved context or the reasoning trace to the user.** If the job is to convey accurate information, expose the sources behind each answer so the user can confirm or reject them. This turns your users into a distributed check on your failure modes.

**Treat user feedback as continuous monitoring.** Log it, track it over time, investigate anomalies, and keep iterating.

**Evaluate offline first, then ship with monitoring.** The ship of certainty sailed long ago. The goal is not to be perfect before deployment, which is unattainable, but to have enough signal to catch a mistake quickly and roll back.

**Stay sceptical of every metric** All of this is probabilistic. No single number tells you the system is good; you are watching whether the whole picture is trending in the right direction. Any metric can be gamed, can drift, or can look healthy while something underneath rots. Trust the convergence of several weak signals over any one strong-looking score.

### Conclusion

There is no clean solution to this. Every evaluation method has a crack, and every crack is filled by some other method you are trusting implicitly. What you can do is be honest about where each layer breaks down, stack layers that fail in different ways, and keep humans in contact with real output often enough to catch what the metrics miss. That is not a perfect answer. It is the only one on offer.

#### References

- Papineni, K., Roukos, S., Ward, T., Zhu, W. (2002). [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/). ACL.
- Lin, C. (2004). [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/). Text Summarization Branches Out, ACL Workshop.
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
- Bowman, S., Angeli, G., Potts, C., Manning, C. (2015). [A Large Annotated Corpus for Learning Natural Language Inference](https://arxiv.org/abs/1508.05326). EMNLP.
- Williams, A., Nangia, N., Bowman, S. (2018). [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426). NAACL.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K., Artzi, Y. (2020). [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675). ICLR.
- Zheng, L., et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS.
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2024). [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217). EACL.
- Panickssery, A., Bowman, S., Feng, S. (2024). [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076). NeurIPS.