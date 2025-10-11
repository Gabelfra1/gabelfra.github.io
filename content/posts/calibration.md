---
author: "Francesco Gabellini"
title: "LLM are way too confident"
date: "2025-10-11"
tags: 
- OpenAI
- Probabilities
---

### The Paradox of High Accuracy and Low Reliability

In this article, we'll explore GPT's impressive ability to achieve high zero-shot performance on a classification tasks: [LLM are multitask learner](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and at the same time being way [overconfident](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on all of their responses.

Calibration it ensures the model "knows what it doesn't know." 
For instance, if a model assigns a probability of 0.8 (or 80% confidence) to a collection of predictions, we expect that 80% of those specific predictions will be correct in reality. When a model is perfectly calibrated, you can directly trust its confidence scores as an honest statement of its likelihood of being right.

GPT can correctly classify a sample without any prior fine-tuning, its confidence scores often don't accurately reflect the likelihood of its predictions being correct. This disparity between performance and reliability is a significant challenge for deploying these models in real-world applications.


#### The data

The dataset for this article is sourced from the readily accessible scikit-learn (sklearn) open datasets. This is a classic NLP classification problem where the raw data consists of various news articles, and the original target variable maps to 20 distinct article topics.

For the sake of simplifying this demonstration, we filtered the original corpus. Our reduced dataset now exclusively comprises articles belonging to three specific categories: 'mac', 'motorcycles', and 'baseball'.


```python 
from sklearn.datasets import fetch_20newsgroups

categories_to_fetch = ['comp.sys.mac.hardware','rec.motorcycles','rec.sport.baseball']
simplified_names = ['mac','motor','baseball']

# Fetch the data, removing headers/footers/quotes for cleaner text
newsgroups_train = fetch_20newsgroups(
    subset='train',
    remove=('headers', 'footers', 'quotes'),
    categories=categories_to_fetch,
    shuffle=True,
    random_state=42
)
newsgroups_test = fetch_20newsgroups(
    subset='test',
    remove=('headers', 'footers', 'quotes'),
    categories=categories_to_fetch,
    shuffle=True,
    random_state=42
)
```

#### The baseline model

Before the AI model, we first need to establish a baseline classifier. This was accomplished using a classical machine learning approach: we vectorized the text data using the well-established TF-IDF (Term Frequency-Inverse Document Frequency) method, and then trained a simple Naive Bayes classifier on the resulting features.

```python 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
```

Our Naive Bayes baseline model achieved a respectable 85% F1 score. 
Crucially, it also demonstrated a decently calibrated output, meaning its predicted probabilities are a reasonably accurate reflection of the true  likelihood for its classifications.

<figure>
  <img src="../../images/baseline_calibration.png" alt="baseline_calibration">
</figure>


#### The AI model

Now we move to a modern approach for this NLP task: zero-shot classification. 
By leveraging the [OpenAI API](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) we completely bypass the need for any training. Instead, we tap directly into the vast pre-training knowledge of GPT and use a prompt to instruct the model to classify the news articles. This method allows us to achieve high performance without a single training loop.

This Python snippet defines a function, get_api_metrics_data, which performs zero-shot classification on a given news article using the OpenAI GPT-4o-mini model. It uses a system prompt to enforce classification into one of three specified categories (simplified_names) and is configured to return log probabilities for the top tokens (logprobs=True, top_logprobs=3) [OpenAI LogProb cookbook](https://cookbook.openai.com/examples/using_logprobs). 

```python 
from openai import OpenAI

# Prompt 
SYSTEM_PROMPT = f"""You are an expert classifier.
Classify the article into exactly one of the following categories: {', '.join(simplified_names)}.
Return ONLY the category name, exactly as it appears in the list, and nothing else."""

# Inference function
def get_api_metrics_data(article):
    """
    Calls the API once and returns both the probability vector (for calibration) 
    and the predicted integer ID.
    """
    prob_vector = np.zeros(3) 
    pred_id = -1

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                      {"role": "user", "content": f"Article: {article}...\nCategory:"}],
            logprobs=True, 
            top_logprobs=3, 
            temperature=0,
        )

        predicted_name = completion.choices[0].message.content.strip().lower()
        pred_id = category_to_id.get(predicted_name, -1)
        logprobs_content = completion.choices[0].logprobs.content
        if logprobs_content:
            first_token_logprobs = logprobs_content[0].top_logprobs
            class_probs = np.zeros(3)
            for lp_item in first_token_logprobs:
                token = lp_item.token.strip().lower()
                linear_prob = np.exp(lp_item.logprob)
                if 'mac' in token or token == 'mac':
                    class_probs[0] += linear_prob
                elif 'motor' in token or token == 'motor':
                    class_probs[1] += linear_prob
                elif 'baseball' in token or token == 'baseball':
                    class_probs[2] += linear_prob
            total_relevant_prob = np.sum(class_probs)
            if total_relevant_prob > 0:
                prob_vector = class_probs / total_relevant_prob
            else:
                if pred_id != -1:
                    prob_vector[pred_id] = 1.0
        return prob_vector, pred_id
    except Exception:
        return np.zeros(3), -1 
```


Our AI model demonstrated an impressive F1 score of 95%, significantly outperforming the Naive Bayes baseline by a solid 10% margin. 
However the model exhibits severe miscalibration, oscillating between being overconfident when its predictions are likely wrong, and being overly pessimistic (underconfident) when its predictions are correct.


<figure>
  <img src="../../images/ai_calibration.png" alt="baseline_calibration">
</figure>

#### Can we fix it ?

To address the observed miscalibration, we applied a traditional post-processing technique: fitting a second regression model to map the estimated scores to true probabilities. Specifically, we leveraged the [Isotinic regression](https://scikit-learn.org/stable/modules/calibration.html) from the scikit-learn package, a common method for recalibrating uncalibrated estimators.

However, isolating the results for a specific class, such as the 'hardware' (mac) category, clearly shows the limitations of this approach:
<figure>
  <img src="../../images/openai_calibration_plot_isotonic_mac.png" alt="baseline_calibration">
</figure>

As the initial confidence scores from the GPT model exhibited virtually no correlation with the real class probabilities, it proved impossible to fix the miscalibration effectively using Isotonic Regression or any other simple linear or non-linear transformation. The underlying issue is the fundamental instability of the model's confidence scores.


#### Why is this a problem ?

In most practical applications, simply providing a prediction label is insufficient; a clear measure of the confidence level in that prediction is absolutely essential. Consider a fraud detection system: it is designed to automatically approve transactions with high-confidence predictions of safety, while low-confidence predictions are deliberately flagged for mandatory human review. If the model's underlying confidence scores are inaccurate or "miscalibrated," this entire risk-managed workflow completely breaks down, leading to either high-risk transactions being overlooked or excessive manual review of safe transactions.


### Conclusion 

While Large Language Models (LLMs) like GPT offer incredible zero-shot performance on NLP tasks, completely eliminating the need for training data, they suffer from a fundamental practical limitation: they currently lack an effective and reliable mechanism for measuring output uncertainty and calibration. 

This absence means that in any use case where decision-making critically depends on a trustworthy measure of confidence or risk, modern LLMs become a terrible choice. Instead, practitioners should rely on simpler, well-calibrated Machine Learning models, even though these require the initial investment of collecting and training on labeled data. The superior performance of the LLM isn't always  worth the cost of unreliable risk assessment.