### Metrics for Generative AI

Generative AI models, such as those for text (e.g., GPT series), images (e.g., DALL-E), or audio, are evaluated using a mix of automated quantitative metrics and qualitative human assessments. These metrics assess aspects like quality, coherence, diversity, fidelity to inputs, and ethical considerations. Below is a table summarizing key metrics commonly used across generative tasks, with descriptions and primary use cases. Note that no single metric is perfect, and combinations (including human evaluation) are often recommended.

---

### **1. Text Generation & Language Modeling**
| Metric | Description | Key Use Cases |
|--------|-------------|---------------|
| **Perplexity** | Measures how well a probability model predicts a sample; lower = better fluency & coherence | Language modeling, next-word prediction |
| **BLEU** | N-gram precision overlap with reference(s); penalizes short outputs | Machine translation, dialogue, text generation |
| **ROUGE** | Recall-oriented n-gram/LCS overlap | Summarization, headline generation |
| **METEOR** | Aligns unigrams with synonyms, stemming, and word order | Translation, paraphrasing |
| **BERTScore** | Cosine similarity of BERT embeddings (semantic) | Any text: faithfulness, QA, summarization |
| **Self-BLEU / Unique n-grams** | Measures diversity by treating one output as "reference" for others | Story generation, open-ended chat |

---

### **2. Image & Visual Generation**
| Metric | Description | Key Use Cases |
|--------|-------------|---------------|
| **FID (Fréchet Inception Distance)** | Compares feature distributions of real vs. generated images | GANs, diffusion models (e.g., Stable Diffusion) |
| **Inception Score (IS)** | Quality + diversity via classifier confidence & entropy | GAN evaluation (legacy; less used now) |
| **Precision & Recall for Distributions** | Separately measures realism (precision) and coverage (recall) | High-res image synthesis |
| **CLIP Score** | Cosine similarity between image and text prompt embeddings | Text-to-image alignment (DALL·E, Midjourney) |

---

### **3. Multimodal & Cross-Modal Tasks**
| Metric | Description | Key Use Cases |
|--------|-------------|---------------|
| **CLIP Score / T5 Score** | Text-image or text-text semantic alignment | Image captioning, visual QA, retrieval |
| **R@K (Recall at K)** | Retrieval accuracy in joint embedding space | Image-text retrieval |
| **Human Preference (Elo, A/B)** | Pairwise human judgments | Text-to-image, video, music |

---

### **4. Safety, Ethics & Fairness**
| Metric | Description | Key Use Cases |
|--------|-------------|---------------|
| **Toxicity Score (Perspective API, RealToxicityPrompts)** | Probability of harmful content | Chatbots, content generation |
| **Bias Metrics (WEAT, CrowS-Pairs, Bias-in-Bios)** | Measures stereotyping in embeddings or outputs | Fairness in hiring, gender/race bias |
| **Regard / Honesty Scores** | Evaluates respectfulness or truthfulness | Dialogue systems, factuality |

---

### **5. General / Human-Centric Evaluation**
| Metric | Description | Key Use Cases |
|--------|-------------|---------------|
| **Human Evaluation (Likert, Ranking, Fluency/Coherence)** | Crowdsourced ratings on multiple axes | **All domains** – gold standard |
| **LLM-as-a-Judge (GPT-4 Eval, Reward Models)** | Uses strong LLM to score outputs vs. references | Scalable alternative to human eval |
| **HELM / BIG-bench / MMLU-style Probes** | Holistic benchmark suites | General capability assessment |

---

### Quick Reference by Task
| Task | Recommended Metrics |
|------|---------------------|
| **Machine Translation** | BLEU, METEOR, BERTScore, chrF |
| **Summarization** | ROUGE, BERTScore, Factuality (e.g., QAGS) |
| **Text-to-Image** | FID, CLIP Score, Human pref |
| **Dialogue / Chat** | Perplexity, Diversity, Toxicity, Human rating |
| **Creative Writing** | Self-BLEU, MAUVE, Human creativity score |
