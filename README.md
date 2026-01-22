# Mental Health Survey Dimension Reduction

## Overview
The goal of this project is to address the challenge of analyzing mental health data collected from heterogeneous surveys, where questions differ in form but often target similar underlying constructs.

In practice, mental health data are collected using many different surveys, and the questions across these surveys are often formulated differently. However, many of these questions may be targeting the same underlying direction or type of mental health construct. Due to differences in question wording and structure, it is difficult to process and analyze data across surveys in a unified way, and this alignment step usually requires substantial manual effort.

To address this issue, we treat each survey question as a representation of an underlying dimension and aim to use machine learning methods to reduce the question space into a smaller number of dimensions. Specifically, we map heterogeneous survey questions into the eight key wellness dimensions defined by Georgia Tech: Emotional, Environmental, Financial, Intellectual, Occupational, Physical, Social, and Spiritual.

Based on this new dimension-level categorization, we then re-aggregate existing patient or user scores from the original survey questions and perform analysis and evaluation using the reduced dimension representation.

## Implementation Steps

This project is implemented in two practical steps, focusing on semantic grouping of survey questions and downstream analysis using re-aggregated scores.

---

### Step 1: Semantic Classification of Survey Questions

Using given mental health survey questions are grouped into the **eight wellness dimensions** using semantic similarity. Each question is encoded using a language model, and machine learning methods are applied to classify questions into one or more of the eight categories based on meaning rather than survey-specific structure.

The output of this step is a mapping from **original survey questions to wellness dimensions**.

#### Possible Implementation Methods

| Method / Model                                   | Accuracy (TBD) | Status       | Description |
|-------------------------------------------------|----------------|--------------|-------------|
| **Prototype-Based Semantic Mapping (Baseline)** | TBD            | Completed    | Use a small set of predefined wellness dimension descriptions as semantic anchors. Each survey question is embedded using a pretrained language model and mapped to the closest dimension prototypes based on cosine similarity. This method requires no labeled training data and provides an interpretable, concept-driven baseline for dimension assignment. |
| Unsupervised Semantic Clustering                 | TBD            | In progress  | Apply unsupervised clustering (e.g., K-means) on question embeddings to discover latent semantic groupings without predefined dimension anchors. Cluster interpretations are analyzed post hoc by comparing cluster contents to the wellness dimensions. This method explores whether natural semantic structure aligns with or diverges from the predefined taxonomy. |
| Supervised Multi-Label Classification            | TBD            | Planned     | Train a supervised multi-label classifier on top of fixed semantic embeddings using a limited set of manually annotated questions. This approach allows each question to be assigned to one or more wellness dimensions and serves as a data-driven refinement over the baseline mapping. |
| LLM Prompt-Based Classification                  | TBD            | Planned     | Use a large language model with prompt-based instructions to assign survey questions to one or more wellness dimensions. This method is used primarily as a qualitative and comparative reference rather than a deployable system. |
---

### Step 2: Dimension-Level Scoring and Analysis

Using given patients scores—originally defined under different survey-specific categories—are re-aggregated according to the **new semantic dimension mapping** from Step 1. Using simple aggregation algorithms, question-level scores are converted into **dimension-level scores** under the eight wellness dimensions.

These dimension-level scores are then used for analysis, such as identifying patterns, comparing subpopulations, or evaluating consistency across surveys.

#### Possible Aggregation and Analysis Methods

| Aggregation / Algorithm | Status |
|-------------------------|--------|
| Sum of scores           | Not started |
| Average score           | Not started |
| Min or Max              | Not started |
| Normalization across dimensions | Not started |
|...|...|
---

## The Eight Dimensions of Wellness (Georgia Tech)

This project adopts the **Eight Dimensions of Wellness framework used by Georgia Tech** to guide the semantic reduction of mental health survey questions. The framework encourages a holistic view of wellness, recognizing that well-being spans emotional, physical, social, and contextual domains, and that individual wellness must be understood in relation to broader environments and systems.

The definitions below are adapted from Georgia Tech materials, informed by work from the Global Wellness Institute, the Substance Abuse and Mental Health Services Administration (SAMHSA), and the University of Maryland at College Park.

| Wellness Dimension | Description |
|------------------|-------------|
| **Emotional** | Coping effectively with life stressors, maintaining self-esteem, expressing optimism, and being aware of, accepting, and appropriately expressing a full range of emotions in oneself and others. |
| **Environmental** | Honoring the dynamic relationship with social, natural, built, and digital environments, and engaging with spaces that are safe, nurturing, stimulating, and sustainable. |
| **Financial** | Meeting basic needs, managing financial resources responsibly, making informed financial decisions, setting realistic financial goals, and preparing for short- and long-term needs or emergencies. |
| **Intellectual** | Engaging in lifelong learning, expanding knowledge and skills, interacting with the world through curiosity and problem-solving, and thinking critically while exploring new ideas. |
| **Occupational** | Deriving personal satisfaction and enrichment from work, study, hobbies, or volunteer activities that align with one’s values, goals, and lifestyle, and taking a proactive approach to career development. |
| **Physical** | Supporting physical health through physical activity, sleep, nutrition, preventive care, and low-risk behaviors related to substance use and overall health maintenance. |
| **Social** | Connecting with others and communities in meaningful ways, maintaining a strong support system, engaging in constructive dialogue, and fostering a sense of belonging, inclusion, and mattering. |
| **Spiritual** | Seeking purpose and meaning in life, practicing self-reflection and gratitude, extending compassion toward others, and cultivating harmony with personal values and the broader world. |

## Datasets
| Dataset | Code | # Questions |
|----------------------------|----------|-------------|
| UCLA Loneliness Scale | UCLA | 20 |
| PERMA Profiler (2016) | PERMA | 23 |
| Psychological Well-Being Scale (18 items) | PWB | 18 |
| Pittsburgh Sleep Quality Index (PSQI) | PSS | 23 |
| Perceived Wellness Survey | PWS | 36 |
| Connor–Davidson Resilience Scale (CD-RISC) | CD_RISC | 25 |
