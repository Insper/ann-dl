### Métricas para IA Generativa

Modelos de IA Generativa, como os de texto (ex: série GPT), imagens (ex: DALL-E) ou áudio, são avaliados usando uma mistura de métricas quantitativas automatizadas e avaliações qualitativas humanas. Essas métricas avaliam aspectos como qualidade, coerência, diversidade, fidelidade às entradas e considerações éticas.

---

### **1. Geração de Texto e Modelagem de Linguagem**
| Métrica | Descrição | Principais Casos de Uso |
|--------|-------------|---------------|
| **Perplexidade** | Mede quão bem um modelo de probabilidade prevê uma amostra; menor = melhor fluência e coerência | Modelagem de linguagem, previsão da próxima palavra |
| **BLEU** | Overlap de precisão de n-gramas com referência(s); penaliza saídas curtas | Tradução automática, diálogo, geração de texto |
| **ROUGE** | Overlap de n-gramas/LCS orientado ao recall | Sumarização, geração de títulos |
| **METEOR** | Alinha unigramas com sinônimos, stemming e ordem de palavras | Tradução, paráfrase |
| **BERTScore** | Similaridade cosseno de embeddings BERT (semântica) | Qualquer texto: fidelidade, QA, sumarização |
| **Self-BLEU / n-gramas únicos** | Mede diversidade tratando uma saída como "referência" para as outras | Geração de histórias, chat aberto |

---

### **2. Geração de Imagens e Visual**
| Métrica | Descrição | Principais Casos de Uso |
|--------|-------------|---------------|
| **FID (Distância de Fréchet Inception)** | Compara distribuições de features de imagens reais vs. geradas | GANs, modelos de difusão (ex: Stable Diffusion) |
| **Inception Score (IS)** | Qualidade + diversidade via confiança e entropia do classificador | Avaliação de GANs (legado; menos usado hoje) |
| **Precisão e Recall para Distribuições** | Mede separadamente realismo (precisão) e cobertura (recall) | Síntese de imagens de alta resolução |
| **CLIP Score** | Similaridade cosseno entre embeddings de imagem e prompt de texto | Alinhamento texto-para-imagem (DALL·E, Midjourney) |

---

### **3. Tarefas Multimodais e Cruzadas**
| Métrica | Descrição | Principais Casos de Uso |
|--------|-------------|---------------|
| **CLIP Score / T5 Score** | Alinhamento semântico texto-imagem ou texto-texto | Legenda de imagem, QA visual, recuperação |
| **R@K (Recall em K)** | Acurácia de recuperação no espaço de embedding conjunto | Recuperação imagem-texto |
| **Preferência Humana (Elo, A/B)** | Julgamentos humanos por pares | Texto-para-imagem, vídeo, música |

---

### **4. Segurança, Ética e Equidade**
| Métrica | Descrição | Principais Casos de Uso |
|--------|-------------|---------------|
| **Pontuação de Toxicidade (Perspective API, RealToxicityPrompts)** | Probabilidade de conteúdo prejudicial | Chatbots, geração de conteúdo |
| **Métricas de Viés (WEAT, CrowS-Pairs, Bias-in-Bios)** | Mede estereótipos em embeddings ou saídas | Equidade em contratação, viés de gênero/raça |
| **Pontuações de Respeito / Honestidade** | Avalia respeitabilidade ou veracidade | Sistemas de diálogo, factualidade |

---

### **5. Avaliação Geral / Centrada no Humano**
| Métrica | Descrição | Principais Casos de Uso |
|--------|-------------|---------------|
| **Avaliação Humana (Likert, Ranking, Fluência/Coerência)** | Avaliações crowdsourced em múltiplos eixos | **Todos os domínios** – padrão ouro |
| **LLM-como-Juiz (GPT-4 Eval, Reward Models)** | Usa LLM poderoso para pontuar saídas vs. referências | Alternativa escalável à avaliação humana |
| **HELM / BIG-bench / Sondas estilo MMLU** | Suítes de benchmark holísticas | Avaliação geral de capacidade |

---

### Referência Rápida por Tarefa
| Tarefa | Métricas Recomendadas |
|------|---------------------|
| **Tradução Automática** | BLEU, METEOR, BERTScore, chrF |
| **Sumarização** | ROUGE, BERTScore, Factualidade (ex: QAGS) |
| **Texto-para-Imagem** | FID, CLIP Score, preferência humana |
| **Diálogo / Chat** | Perplexidade, Diversidade, Toxicidade, avaliação humana |
| **Escrita Criativa** | Self-BLEU, MAUVE, pontuação de criatividade humana |


---

--8<-- "docs/2026.2/classes/metrics/generative/quiz.pt.md"
