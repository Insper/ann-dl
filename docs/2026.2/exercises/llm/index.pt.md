!!! success inline end "Prazo e Entrega"

    :date: A definir
    
    :clock1: Commits até 23:59

    :material-account-group: Equipe (2–3 membros)

    :simple-github: Link do GitHub Pages via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Atividade: Fine-Tuning de um Large Language Model com LoRA**

Nesta atividade você irá fazer fine-tuning de um LLM pré-treinado em uma tarefa personalizada usando **LoRA (Low-Rank Adaptation)**, uma das técnicas de Fine-Tuning com Eficiência de Parâmetros (PEFT) mais amplamente usadas na indústria. Você usará o ecossistema Hugging Face: `transformers`, `peft` e `trl`.

---

## Objetivos de Aprendizagem

Ao final desta atividade você será capaz de:

1. Carregar e inspecionar um LLM pré-treinado e seu tokenizador
2. Configurar e aplicar adaptadores LoRA a módulos de atenção específicos
3. Fazer fine-tuning do modelo em um dataset de instruções personalizado usando `SFTTrainer`
4. Avaliar o modelo com fine-tuning qualitativa e quantitativamente
5. Entender o tradeoff entre parâmetros treináveis, memória e qualidade

---

## Pré-requisitos

Instale os pacotes necessários:

```bash
pip install transformers peft trl datasets accelerate bitsandbytes
```

Você precisará de acesso a uma GPU (Google Colab Pro ou similar). Para um experimento mínimo, `microsoft/Phi-3-mini-4k-instruct` (3,8B parâmetros) ou `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1,1B) funcionam em uma GPU T4.

---

## Exercício 1 — Inspeção do Modelo e Baseline

### Instruções

1. **Carregue um modelo pré-treinado** e seu tokenizador:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
```

2. **Conte o total de parâmetros** e reporte:
   - Total de parâmetros
   - Parâmetros por tipo de camada (embedding, atenção, FFN)
   - Uso de memória (use `model.get_memory_footprint()`)

3. **Execute uma inferência baseline** com 5 prompts relacionados à sua tarefa escolhida (ex: análise de sentimento, Q&A, geração de código). Registre as respostas brutas do modelo.

4. **Avalie o baseline** usando uma métrica apropriada para sua tarefa (acurácia para classificação, BLEU/ROUGE para geração). Isso estabelece o baseline antes do fine-tuning.

---

## Exercício 2 — Preparação do Dataset

### Instruções

1. **Escolha ou crie um dataset de tarefa** com pelo menos 500 exemplos de seguimento de instruções. Fontes sugeridas:
   - Hugging Face Datasets (ex: `financial_phrasebank`, `medical_questions_pairs`)
   - Dataset personalizado relevante ao seu domínio

2. **Formate no formato de seguimento de instruções** (compatível com o template de chat do modelo):

```python
def formatar_exemplo(example):
    return {
        "text": tokenizer.apply_chat_template([
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ], tokenize=False, add_generation_prompt=False)
    }
```

3. **Divida** 80% treino / 10% validação / 10% teste. Reporte:
   - Tamanho do dataset, distribuição de rótulos (se classificação)
   - Comprimento médio de entrada/saída em tokens
   - Exemplo de uma amostra de treinamento formatada

---

## Exercício 3 — Configuração de LoRA e Fine-Tuning

### Instruções

1. **Configure o LoRA** usando `peft`:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                         # rank — experimente com 4, 8, 16
    lora_alpha=16,               # fator de escalonamento (tipicamente 2×r)
    target_modules=["q_proj", "v_proj"],  # módulos a adaptar
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

2. **Faça fine-tuning com SFTTrainer** (código conforme a documentação do Hugging Face).

3. **Execute experimentos** com pelo menos **duas configurações diferentes de LoRA** e compare:
   - `r=4` vs `r=16`
   - `target_modules=["q_proj", "v_proj"]` vs `["q_proj", "k_proj", "v_proj", "o_proj"]`

4. **Plote curvas de perda de treinamento e validação** para cada configuração.

### Tabela obrigatória de resultados

| Config | Rank r | Módulos alvo | Params treináveis | Perda treino | Perda val | Métrica teste |
|--------|--------|---------------|-----------------|-----------|---------|------------|
| A | 4 | q, v | ? | ? | ? | ? |
| B | 16 | q, k, v, o | ? | ? | ? | ? |

---

## Exercício 4 — Avaliação e Análise

### Instruções

1. **Avaliação quantitativa** no conjunto de teste:
   - Classificação: acurácia, F1
   - Geração: BLEU-4, ROUGE-L, ou métrica específica da tarefa
   - Compare baseline vs. cada configuração LoRA

2. **Avaliação qualitativa**: execute 10 prompts de teste no baseline e no melhor modelo com fine-tuning. Mostre 3 exemplos onde o fine-tuning claramente melhorou a saída e 1–2 onde não melhorou.

3. **Análise de erros**: com quais tipos de entrada o modelo com fine-tuning ainda tem dificuldades? É um problema de dados, de prompt ou de capacidade?

4. **Ablação — análise de rank**: se possível, teste `r ∈ {2, 4, 8, 16, 32}` e plote a métrica de teste vs. contagem de parâmetros treináveis. Em que rank o desempenho estabiliza?

---

## Exercício 5 — Reflexão

Responda às seguintes perguntas em seu relatório (1 parágrafo cada):

1. Quantos parâmetros o LoRA realmente treinou? Qual percentual do modelo completo? Por que isso é suficiente para adaptar o modelo à sua tarefa?

2. O que o fine-tuning completo teria exigido em termos de memória? Por que isso é proibitivo para a maioria das equipes?

3. O modelo com fine-tuning "esqueceu" alguma capacidade geral do modelo base? Forneça evidências de sua avaliação qualitativa.

4. Se você fosse implantar este modelo com fine-tuning em produção, quais etapas adicionais você tomaria antes do lançamento?

---

## Critérios de Avaliação

!!! danger "Restrições Importantes"
    - Use **apenas modelos open-source** (sem API do GPT-4, sem API do Claude). O modelo deve ser carregável do Hugging Face Hub.
    - Reporte as **horas de GPU usadas** e o custo estimado.
    - Todo código deve ser reprodutível: defina sementes aleatórias, fixe versões de bibliotecas.

| Critério | Pontos |
|:---:|---|
| **2 pts** | Seleção, preparação e formatação do dataset |
| **2 pts** | Configuração do LoRA e fine-tuning bem-sucedido |
| **2 pts** | Avaliação quantitativa e comparação de configurações |
| **2 pts** | Avaliação qualitativa e análise de erros |
| **2 pts** | Perguntas de reflexão e qualidade do relatório |

**Formato de entrega:** Relatório no GitHub Pages + link para o notebook de treinamento (Google Colab ou similar). Inclua todos os plots e tabelas.

**Colaboração com IA:** Permitida, mas você deve entender cada parâmetro de configuração. O relatório deve ser sua própria análise.
