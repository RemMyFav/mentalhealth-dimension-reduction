# Mental Health Survey Dimension Reduction

## Overview
The goal of this project is to address the challenge of analyzing mental health data collected from heterogeneous surveys, where questions differ in form but often target similar underlying constructs.

In practice, mental health data are collected using many different surveys, and the questions across these surveys are often formulated differently. However, many of these questions may be targeting the same underlying direction or type of mental health construct. Due to differences in question wording and structure, it is difficult to process and analyze data across surveys in a unified way, and this alignment step usually requires substantial manual effort.

To address this issue, we treat each survey question as a representation of an underlying dimension and aim to use machine learning methods to reduce the question space into a smaller number of dimensions. Specifically, we map heterogeneous survey questions into the eight key wellness dimensions defined by Georgia Tech: Emotional, Environmental, Financial, Intellectual, Occupational, Physical, Social, and Spiritual.

Based on this new dimension-level categorization, we then re-aggregate existing patient or user scores from the original survey questions and perform analysis and evaluation using the reduced dimension representation.

## Implementation Steps

This project is implemented in two practical stages:

1. **Semantic Classification of Survey Questions**, including comparison with human annotations to evaluate reliability and variability.

2. **Generated Question Classification into the Eight Wellness Dimensions**, exploring LLM-based question generation and evaluating classification behavior under the same semantic mapping framework.
---

### Step 1: Semantic Classification of Survey Questions

In this step, heterogeneous mental health survey questions are grouped into the eight wellness dimensions using semantic similarity.

Instead of relying on a single fixed set of dimension definitions, we generate five independent definition sets for the eight wellness dimensions using different large language models (LLMs). Each definition set provides a slightly different natural-language description of the same conceptual dimensions.

All survey questions are encoded using a pretrained sentence embedding model. For each definition set:
	•	The eight dimension definitions are embedded.
	•	Each survey question is embedded.
	•	Cosine similarity is computed between the question embedding and each dimension definition embedding.
	•	Questions are assigned to dimensions using a margin-based selection rule.

This process is repeated across all five LLM-generated definition sets. By comparing assignments across definition sets, we evaluate:
	•	Robustness of dimension mapping under definition variation
	•	Agreement levels across LLM-generated anchors
	•	Consistency relative to human annotations

The output of this step includes:
	•	A mapping from original survey questions to wellness dimensions
	•	Cross-LLM agreement statistics
	•	Human vs LLM comparison metrics (e.g., JSD, entropy)

### Step 2: Dimension-Level Scoring and Analysis
Step 2: Generated Question Classification and Evaluation

In addition to classifying existing survey questions, we explore LLM-based generation of new mental health self-report items.

We experiment with two generation strategies:
	•	Pure prompt-based generation
	•	Anchor-guided generation (sampling existing items as semantic seeds)

Generated items are then passed through the same semantic classification pipeline described in Step 1. This allows us to:
	•	Evaluate whether generated items map coherently into the eight wellness dimensions
	•	Measure classification stability under cosine-based mapping
	•	Explore quantitative criteria for defining a “good” survey question

Together, these two steps create a unified framework for:
	•	Semantic alignment of heterogeneous survey questions
	•	Reliability analysis (internal and population-level)
	•	Controlled generation and evaluation of new items

## Project Structure
```text
.
├── src/                   # Core reusable modules (embedding, similarity, mapping, selection)
│
├── labtory/               # Experimental workflows (reproducible experiments)
│   ├── internal_jsd/
│   │   ├── input/
│   │   │   ├── human_annotations.csv
│   │   │   └── repeated_items.csv
│   │   ├── temp/
│   │   │   └── sampled_subsets.pkl
│   │   ├── output/
│   │   │   ├── jsd_scores.csv
│   │   │   └── jsd_plot.png
│   │   └── notebooks/
│   │       └── analysis_internal_jsd.ipynb
│   │
│   ├── population_entropy/
│   ├── mapping_robustness/
│   ├── clustering_ud/
│   └── generation/
│
├── logs/                  # Experiment logs and diagnostics
│
├── README.md
├── requirements.txt
└── .venv/
```

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

