# Sprint 02 Report — Internal vs Population-Level Consistency Analysis

## 1. Objectives

This sprint focused on systematically comparing internal consistency (within repeated measurements) and population-level dispersion (across annotators or model variants) to better understand the reliability and stability of the dimension assignment pipeline.

In addition, this sprint included initial exploratory work on question generation and a structural refactoring of the whole github codebase.


### **Internal Consistency (within repeated items)**

- Use **Jensen–Shannon Divergence (JSD)** to quantify inconsistency in repeated human-labeled questions (test–retest reliability).
- Apply the same repeated questions to our pipeline:
  - Use the 5 independently generated LLM definition sets (8 dimensions each).
  - Perform cosine similarity mapping under each definition set.
  - Examine consistency of assignments under a single LLM across definition variations.
- Hypothesis:
  - LLM outputs should be fully consistent for identical repeated inputs (JSD ≈ 0).
  - Human annotations are expected to exhibit non-zero JSD due to annotator variability.

---

### **Population-Level Dispersion**

- Use **normalized entropy** to measure disagreement across annotators for non-repeated questions.
- Apply the same entropy-based analysis to LLM outputs:
  - Treat the 5 definition-set runs as a 5-member committee.
  - Compute entropy of assigned dimensions per question.
- Use **down-sampling (matched 5 vs 5)** to ensure fair comparison between:
  - Human annotator variability
  - LLM definition-set variability
- Compare dispersion magnitudes to assess whether LLM variability falls within human-level disagreement.

---

### **Question Generation**

- Conduct initial experiments with LLM-based question generation.
- Explore strategies to obtain:
  - Stable,
  - high-quality,
  - semantically diverse questions.
- Evaluate generated questions using the existing semantic mapping pipeline.
- Begin exploring principled quantitative criteria for defining what constitutes a “good” generated question.

---

### **Codebase Refactoring**

- Reconstruct the entire workspace for improved modularity and reproducibility.

- Goals of refactoring:
  - Improve readability.
  - Separate implementation from experimentation.
  - Ensure reproducibility for:
    - internal JSD analysis,
    - population-level entropy analysis,
    - semantic mapping experiments,
    - question generation experiments.

## 2. Completed Work

### 2.1 Internal Consistency (within repeated items)

To evaluate internal stability, we measured Jensen–Shannon Divergence (JSD) on repeated (test–retest) items.
(See internal_jsd.png)
For human annotations:

- Repeated questions were split into repetition-1 and repetition-2.
- For each run:
  - 5 annotators were sampled (k_sources = 5).
  - The sampling process was repeated 100 times to stabilize estimates.
- JSD was computed between repetition-1 and repetition-2 label distributions.

**Human internal consistency results (k = 5, 100 runs):**

- Mean JSD: **0.0716**
- Median JSD: **0.0654**
- 10th percentile: **0.0367**
- 90th percentile: **0.1134**

This indicates measurable internal variability in human judgments, even for identical repeated questions.

For LLM-based mapping:

- The same repeated questions were mapped using:
  - 5 independently generated LLM definition sets.
  - Cosine similarity mapping under each definition set.
- The same sampling schedule (k = 5, 100 runs) was applied.
- JSD was computed across repeated inputs.

**LLM internal consistency results:**

- Mean JSD: **0.000**
- Median JSD: **0.000**
- All percentiles: **0.000**

This confirms that, under fixed embeddings and cosine similarity mapping, the LLM-based pipeline is fully deterministic and produces perfectly stable outputs for identical repeated inputs.

**Interpretation:**

- Human annotations exhibit intrinsic internal variability (≈0.07 average JSD).
- The LLM mapping pipeline shows perfect internal stability (JSD = 0).
- Therefore, part of observed dispersion in label assignments is attributable to inherent human judgment variability rather than model instability.

### 2.2 Population-Level Dispersion

To evaluate cross-source disagreement, we measured **normalized entropy** across non-repeated questions.
(See populational_jsd.png)
For human annotations:

- For each run:
  - 5 annotators were randomly sampled (k_sources = 5).
  - Sampling was repeated 100 times to stabilize estimates.
- Entropy was computed per question and averaged across questions to obtain an overall dispersion score.

**Human population-level dispersion results (k = 5, 100 runs):**

- Mean entropy: **0.2446**
- Median entropy: **0.2328**
- 10th percentile: **0.1869**
- 90th percentile: **0.3176**

Human entropy varies across runs due to random annotator sampling, reflecting natural inter-annotator disagreement.

For LLM-based mapping:

- The 5 independently generated dimension definition sets were treated as a fixed 5-member ensemble.
- For each question:
  - Dimension assignments across the 5 definition sets were collected.
  - Entropy was computed in the same way as for humans.
- The same k = 5 schedule was applied (but no subsampling was needed since the ensemble is fixed).

**LLM population-level dispersion results:**

- Mean entropy: **0.2443**
- Median entropy: **0.2443**
- All percentiles: **0.2443** (constant across runs)

The LLM entropy remains constant because the same 5-definition ensemble is used in every run.

**Interpretation:**

- The mean entropy of humans and LLMs is nearly identical (≈0.245).
- Human entropy varies due to sampling variability.
- LLM entropy is stable due to a fixed ensemble.

This suggests that cross-definition disagreement in LLM mapping is comparable in magnitude to cross-annotator disagreement among humans at k = 5 sources.

### 2.3 Question Generation
For the generation component, two strategies were explored:

**(A) Pure prompt-based generation**

- The model is prompted to directly generate self-report statements for a 1–5 agreement scale.
- This approach is simple and flexible.
- However, outputs were found to be sensitive to:
  - Prompt phrasing,
  - Decoding parameters (e.g., temperature, top-p),
  - Minor changes in instruction wording.
- As a result, question quality, tone, and dimensional focus can vary noticeably across runs, making stability difficult to guarantee.

Promot:
```python
    return f"""
    Write one self-report mental health statement suitable for a 1–5 agreement scale.

    The statement should:
    - Be written in first person (e.g., "I ...").
    - Be a single clear declarative sentence.
    - Describe a stable feeling, belief, tendency, or life pattern.
    - Sound like a psychological assessment item.
    - Avoid clinical diagnoses or medical terms.
    - Not mention surveys, questionnaires, or instructions.

    Output only the statement.
    """
```
---

**(B) Anchor-guided generation (current preferred approach)**

- A small set of existing questionnaire items is included in the prompt as semantic anchors.
- The model is asked to generate a new self-report statement that remains semantically aligned with the anchor set.
- To prevent the output from being overly tied to a single item:
  - Five anchor items are randomly sampled for each generation run.
  - The model is instructed to blend or fuse these items into a coherent new statement.

This strategy has several advantages:

- Produces more stable and semantically grounded outputs.
- Maintains diversity across wellness dimensions due to random anchor sampling.
- Reduces sensitivity to prompt micro-variations.
- Does not require large-scale models to achieve acceptable quality.

Overall, anchor-guided generation provides a balance between semantic control and diversity, making it a promising direction for further refinement.

Promot:
```python
    return f"""
    Based on the following examples, create a NEW self-report mental health statement.

    Do NOT rewrite or paraphrase any example.
    The new statement must be conceptually related but linguistically distinct.

    Requirements:
    - One clear declarative sentence
    - Natural and realistic
    - Do not copy phrases from the originals

    Original examples:
    {seeds_text}
    """
```
### 2.4 Codebase Refactoring

To improve clarity, modularity, and experimental reproducibility, the entire workspace was restructured.

#### Overall Structure

- `src/`
  - Contains all reusable functional modules and class definitions.
  - Serves as the core implementation layer of the project.
  - Includes components such as:
    - Embedding utilities
    - Similarity computation
    - Margin-based selection
    - Semantic mapping logic
  - Designed to remain experiment-agnostic and reusable across studies.

- `labtory/`
  - Contains all key experimental workflows.
  - Each major experiment is organized into a dedicated folder.
  - Serves as the experimental layer built on top of `src/`.

#### Standardized Experiment Folder Structure

Each experiment follows a consistent structure to ensure readability and reproducibility:

- `input/`
  - Raw or preprocessed input files required for the experiment.

- `output/`
  - Final result files (e.g., CSV summaries, processed outputs).

- `temp/`
  - Intermediate artifacts, cached files, or debugging outputs.

- `notebooks/` (ipynb)
  - Exploratory or visualization notebooks associated with the experiment.

## 3. Open Questions and Challenges

### 3.1 How to Define a “Good” Question?

A key open question is how to define a clear and measurable criterion for question quality.

Currently, quality is judged mostly by intuition (clarity, coherence, dimensional relevance). However, for systematic evaluation and improvement, we need a quantitative measure.

Questions to consider:

- Should a good question have low ambiguity (low entropy under mapping)?
- Should it strongly align with one dimension (high cosine margin)?
- Should it avoid being overly generic?
- Should structural simplicity (e.g., single clear clause) be considered?

We need a practical metric that balances clarity, stability, and informativeness.

---

### 3.2 Self-Loop Generation

If a reliable quality metric can be defined, generation could be improved iteratively:

- Generate candidate questions.
- Score them using the quality metric.
- Keep higher-quality questions.
- Use them to guide further generation.

Challenges:

- Avoid collapsing into repetitive patterns.
- Maintain diversity across dimensions.
- Ensure improvement reflects real quality, not just optimization to the metric.

---

### 3.3 Multi-Dimensional Survey Handling

Wellness constructs are inherently multi-dimensional, but current human annotations are single-label.

Open questions:

- How should multi-dimensional questions be evaluated?
- Should classification remain single-label for consistency?
- How can multi-label assignment be implemented and validated?
- Is new multi-label human annotation needed?

Handling multi-dimensional structure properly is important for improving realism and validity.

