---
author: "Francesco Gabellini"
title: "PageIndex: When Your RAG Reads Like a Human"
date: "2026-05-12"
tags: 
- LLM
- RAG
---

### The Explainability Problem Nobody Talks About

Standard RAG has a transparency problem, and it's not one that shows up in benchmark numbers.

When a user asks *"why did the system return this paragraph and not that one?"*, the honest answer is: *"because the cosine similarity was 0.83 instead of 0.79."* That answer is technically correct and completely useless. Nobody outside of an ML team reasons about document retrieval in terms of vector distances. They reason about sections, chapters, topics, and arguments.

This gap between how retrieval systems work and how humans think about information lookup creates a problem: the users either accept the result blindly or reject the whole system because they can't audit it. Neither is good.

[PageIndex](https://github.com/VectifyAI/PageIndex) is an approach that sidesteps this problem entirely by discarding vector similarity and replacing it with something more explainable : **structure-aware, reasoning-based retrieval**.

---

### How Humans Actually Look Things Up

If you remember searching for something in a physical library before Google, I do, you remember the process intuitively. You didn't scan every page of every book. You:

1. Found the right book by subject
2. Opened the table of contents
3. Found the relevant chapter
4. Skimmed the section headings
5. Read the specific paragraph

That multi-level navigation was precise and fully explainable. You could tell anyone exactly why you landed on page 247 of *Option futures and other derivatives*.

PageIndex replicates this exact behavior. Instead of embedding chunks and retrieving by similarity, it first builds a hierarchical index of the documents. Baiscally a machine-readable table of contents with summaries at each node and then asks an LLM to *reason* about which branch of that tree is likely to contain the answer. The LLM walks the index the same way I used to serarch the uni library.

---

### Two Phases: Index Then Retrieve

The workflow has two clean phases.

**Phase 1 — Build the index.** Submit a document to PageIndex. 
It parses the document structure (headings, sections, subsections), generates a summary for each node, and returns a JSON tree. This happens once per document. 

**Phase 2 — Reason and retrieve.** At query time, pass the tree (without full text) to an LLM and ask it to identify which nodes are relevant. Then pull the actual text only from those nodes and generate the final answer.

The tree for a typical long document looks like this:

```
Document Root
├── Abstract [pages 1-1]
│   └── "Overview of the proposed method and key results..."
├── Introduction [pages 2-3]
│   └── "Motivation, problem statement, and prior work..."
├── Methodology [pages 4-8]
│   ├── Data Collection [pages 4-5]
│   ├── Model Architecture [pages 5-7]
│   └── Training Procedure [pages 7-8]
├── Results [pages 9-12]
│   ├── Quantitative Evaluation [pages 9-11]
└── Conclusion [pages 11-12]
    └── "Summary of findings and future directions..."
```

The LLM sees this structure with the summaries and can reason: *"the question is about conclusions, so I should look at the Conclusion node."* That reasoning is visible, auditable, and directly explainable to any user.

---

### The Code

Setup is minimal : Clone, install dependencies, and set your LLM key. 
Then generate the tree locally from a PDF :

```bash
python3 run_pageindex.py --pdf_path document.pdf \
    --if-add-node-summary yes \
    --if-add-node-text yes
```

Load the result and set up your LLM client:

```python
import pageindex.utils as utils
import openai, json

with open("results/document_structure.json") as f:
    tree = json.load(f)

utils.print_tree(tree)

def call_llm(prompt, model="gpt-4.1", temperature=0):
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
```

At query time, strip the full text out of the tree, show it to the LLM, and ask for the relevant nodes:

```python
query = "What are the main conclusions?"

tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, title, and summary.
Find all nodes likely to contain the answer.

Question: {query}
Document tree: {json.dumps(tree_without_text, indent=2)}

Reply in JSON:
{{
    "thinking": "<your reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""

tree_search_result = call_llm(search_prompt)
```

Then extract text only from the selected nodes and generate the answer:

```python
node_map = utils.create_node_mapping(tree)
result = json.loads(tree_search_result)

node_list = result["node_list"]
relevant_content = "\n\n".join(node_map[nid]["text"] for nid in node_list)

answer = await call_llm(f"Answer based on context:\n\nQuestion: {query}\nContext: {relevant_content}")
```

---

### Stateless by Design, Storable if Needed

Because the index is just JSON, the whole system is flexible about state. You can run it entirely in memory for small queries, or persist the tree to disk, a database, or an object store, you do you.

```python
import json

# persist
with open("doc_tree.json", "w") as f:
    json.dump(tree, f)

# restore
with open("doc_tree.json") as f:
    tree = json.load(f)
```

This is meaningful in practice. Vector databases introduce a whole infrastructure layer: embedding models, database extensions, fancy fusion algorithm. PageIndex's index is a JSON file. It integrates with whatever you already have,an S3 bucket, a Redis cache,just some memory. No additional infrastructure required.

---

### Structure-Aware Chunking vs Fixed-Size Chunking

This is perhaps the most underappreciated difference.

Traditional RAG cuts documents into fixed-size chunks, typically 512 or 1024 tokens, with some overlap to avoid losing context. The problem is that documents are not uniformly structured. A single section might be 200 tokens; another might be 3000. A fixed-size chunk will routinely split a coherent argument mid-sentence or merge the end of one section with the start of an unrelated one.

```
Traditional RAG chunking (fixed-size):
│◄── 512 tokens ──►│◄── 512 tokens ──►│◄── 512 tokens ──►│
 [intro...][ch1 end][ch2 start...mid  ][...ch2 end][ch3...]

PageIndex chunking (structure-aware):
│◄── Introduction ──►│◄─────── Chapter 2 ─────────►│◄── Chapter 3 ──►│
 [coherent unit       ] [coherent unit               ] [coherent unit   ]
```

Structure-aware chunks are semantically complete. When shown to the user as a source reference, they read like a passage — not like a sentence that got cut off because a token counter hit a limit.

---

### Scaling Limitations

PageIndex is not a silver bullet. It is fundamentally an in-context approach: the entire tree (without text) is passed to the LLM at query time. For very long documents with deep, granular structure, this tree can itself become large. The LLM must process it in a single context window.

It is worth being honest about this. But it is equally worth being honest  about the fact that **standard RAG also does not excel at scale**. Top-k retrieval over large corpora produces noisy results; relevance degrades as the corpus grows and context bloating is real. 

For the use cases where it fits: long structured documents, regulatory filings, technical manuals, research papers, contracts PageIndex is genuinely strong. [FinanceBench results](https://github.com/VectifyAI/PageIndex) report ~98.7% accuracy on financial document QA.

---

### Conclusion

PageIndex is a good example of a retrieval approach that optimises for the right thing. Not just accuracy, but **explainability**. The retrieval path is a reasoning trace. The chunks are coherent sections. The index is a plain JSON file.

None of this is magic. It is essentially the same thing a well-organised student does when they open a textbook. The insight is just recognising that human information retrieval is already an excellent algorithm, and that replicating it is could be better than replacing it with cosine similarity.

The official repo and cookbook are at [github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) if you want to run the full notebook.

As a recovering statistician, I am profoundly happy when I can make an AI system a bit more explainable, and profoundly unhappy when I can't.
