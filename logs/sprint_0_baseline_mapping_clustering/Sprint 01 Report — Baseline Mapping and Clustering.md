# Sprint 01 Report — Baseline Mapping and Clustering

## 1. Objectives

The primary objective of this sprint was understanding the **conceptual grounding** 

Specifically, the minimum goals were:
- To **understand the structure and content** of the mental health survey datasets involved (e.g., PSQI, PSS, PWB, CD-RISC, UCLA Loneliness, PERMA)
- To **clarify the final task**: reducing heterogeneous survey questions into a shared set of wellness dimensions that cut across instruments

This sprint was intended to establish a solid foundation before introducing more complex modeling or evaluation strategies.

---

## 2. Completed Work

During this sprint, a complete **baseline pipeline** was implemented and validated through qualitative inspection.

Key accomplishments include:

- **Semantic Mapping Baseline**
  - Adopted **Georgia Tech’s official 8 Dimensions of Wellness definitions** as the baseline semantic prototypes (Emotional, Physical, Social, Spiritual, Intellectual, Occupational, Environmental, Financial)
  - Used a pretrained sentence encoder to embed both survey questions and the GT dimension definitions into a shared semantic space
  - Assigned each question to the most similar dimension using cosine similarity
  - Introduced a **margin-based rule** to allow limited top-2 multi-label assignments when similarity scores were close
  - Detailed results and intermediate outputs for this sprint are available in:
  
    \- `s1_kmeans8_cluster.csv` (unsupervised clustering baseline)
  
- **Unsupervised Clustering Baseline**
  - Applied KMeans clustering (K=8) directly on question embeddings
  
  - Extracted cluster-level representatives and similarity-to-center scores

  - Enabled qualitative comparison between clustering structure and semantic mapping results
  
  - Detailed results and intermediate outputs for this sprint are available in:
  
    \- `s1_kmeans8_cluster.csv` (unsupervised clustering baseline)
  
- **Data & Pipeline Infrastructure**
  - Built a unified data pipeline for loading, embedding, mapping, and clustering across all datasets
  - Exported structured CSV outputs for both mapping and clustering results
  - Organized outputs under a consistent results directory for reproducibility

- **Project Documentation**
  - Created and structured a GitHub repository documenting:
    - Project motivation and scope
    - Baseline methods and pipeline
    - Planned follow-up methods (supervised and LLM-based)
  - Added visualizations and example outputs to support qualitative analysis

Overall, this sprint successfully transformed the project from an abstract idea into a **working, inspectable, and extensible research pipeline**.

---

## 3. Open Questions and Challenges

Despite the progress, several key challenges remain unresolved:

- **Lack of Ground Truth**
  - There is currently no clear, authoritative ground truth for assigning survey questions to the proposed wellness dimensions
  - Existing instruments were not designed around this shared dimensional framework, making direct quantitative evaluation difficult

- **Validity of GT Definition Prototypes**
  - The semantic mapping approach treats the **GT wellness definitions** as embedding prototypes (anchors)
  - It remains unclear whether these definitions are:
    - Sufficiently specific to act as reliable semantic anchors across instruments, or
    - Too broad, causing ambiguous assignments (especially for Environmental / Financial)

- **Evaluation Strategy**
  - Without ground truth labels, it is uncertain how best to:
    - Measure accuracy or consistency
    - Compare mapping vs. clustering outcomes beyond qualitative inspection

