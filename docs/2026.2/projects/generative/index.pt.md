
!!! success inline end "Prazo e Entrega"

    :date: A definir
    
    :clock1: Commits até 23:59

    :material-account-group: [Equipe (2–3 membros)](){ :target="_blank" }

    :simple-github: Link do GitHub Pages via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

Este é um projeto de tema livre onde você explora **modelos generativos modernos**. Você deve usar pelo menos uma arquitetura da lista abaixo e construir um pipeline de geração completo. O foco está em compreender os modelos subjacentes, não apenas executar demos — você deve explicar a arquitetura, as conexões entre componentes e as escolhas de design.

## Arquiteturas Elegíveis

Escolha **pelo menos uma** das seguintes como seu modelo principal:

| Trilha | Família de modelos | Exemplos |
|-------|-------------|---------|
| **A** | Difusão Latente + U-Net | Stable Diffusion 1.5/XL |
| **B** | Flow Matching + DiT | FLUX.1-dev, SD3 |
| **C** | Geração Autorregressiva de Imagens | LlamaGen, MaskGIT |
| **D** | Geração de Vídeo | CogVideoX, AnimateDiff |
| **E** | Geração de Áudio | Stable Audio, AudioCraft |
| **F** | Multimodal de qualquer-para-qualquer | Variantes open-source do Chameleon |

!!! tip "Ponto de partida recomendado"
    Trilha B (FLUX.1) ou Trilha A (Stable Diffusion XL via ComfyUI) são as mais acessíveis enquanto cobrem o maior conteúdo do curso. Trilha D ou E são excelentes se sua equipe quiser ir além.

## Requisitos do Pipeline

Seu pipeline deve encadear **pelo menos dois componentes de modelo**. Exemplos:

- Texto → Codificador CLIP → FLUX DiT (Flow Matching) → Decoder VAE → Imagem
- Imagem → Estimador de profundidade → ControlNet + SD → Imagem estilizada
- Texto → LLM (prompt aprimorado) → FLUX → Imagem → Gerador de legenda BLIP → prompt refinado
- Áudio → Whisper → LLM → TTS → novo áudio
- Texto → LLM cria história → SD imagem por cena → vídeo montado

## O que Você Deve Explicar

Para **cada componente de modelo** em seu pipeline, seu relatório deve descrever:

1. **Arquitetura**: que tipo de rede (U-Net, DiT, Transformer, VAR…), número de parâmetros, escolhas de design chave
2. **Objetivo de treinamento**: perda de difusão, flow matching, contrastivo, autorregressivo, etc.
3. **Papel no pipeline**: qual entrada recebe, qual saída produz, por que este componente está aqui
4. **Conexão com o conteúdo do curso**: vincule explicitamente à aula relevante (ex: "Esta U-Net usa atenção cruzada conforme descrito na aula de Atenção")

!!! danger "Sem Almoço Grátis"
    Use apenas modelos open-source e computação gratuita (Google Colab, Kaggle, Hugging Face Spaces). Não use APIs pagas (OpenAI, Midjourney, Adobe Firefly). Documente as horas de GPU usadas.

## Exemplos de Pares Entrada–Saída

Forneça **pelo menos 8 exemplos** mostrando:

- Diferentes prompts de texto / estilos de entrada
- Diferentes parâmetros de inferência (escala CFG, número de passos, semente)
- Pelo menos 2 casos de falha com análise de por que falharam

## Critérios

| Critério | Descrição |
|:---------:|-------------|
| **I** | Entrega incompleta ou sem explicação da arquitetura. |
| **D** | Pipeline básico funcionando com erros; explicação da arquitetura ausente ou superficial. |
| **C** | Um pipeline funcionando (Trilha A ou B) com explicação completa da arquitetura para cada componente. Pelo menos 8 exemplos de entrada-saída com parâmetros variados. |
| **B** | Dois pipelines funcionando ou um pipeline com técnicas avançadas (ControlNet, IP-Adapter, fine-tuning LoRA ou geração de vídeo). Documentação completa da arquitetura. |
| **A** | Nota B mais: fine-tuning personalizado (LoRA/DreamBooth), pipeline original combinando ≥3 componentes, ou implementação de Trilha D/E. Resultados benchmarkados (FID, CLIP Score ou métrica específica do domínio). |

Meio ponto será adicionado ou subtraído com base na qualidade do relatório, criatividade e profundidade da análise arquitetural.

## Estrutura do Relatório

Seu relatório no GitHub Pages deve incluir:

1. **Introdução**: qual pipeline você construiu e por que o escolheu
2. **Diagramas de arquitetura**: diagramas de fluxo mostrando o fluxo de dados entre componentes (use Mermaid ou draw.io)
3. **Análises detalhadas dos componentes**: uma seção por componente com descrição da arquitetura e matemática onde relevante
4. **Galeria de resultados**: pares de entrada-saída anotados com configurações de parâmetros
5. **Análise de falhas**: o que não funciona e por quê
6. **Reflexão**: o que você aprendeu, o que te surpreendeu, o que faria diferente

!!! example "Exemplo de Diagrama de Arquitetura"
    ```mermaid
    flowchart LR
        A["Prompt de Texto"] --> B["Codificador de Texto CLIP\n(ViT-L/14, 123M params)"]
        N["Ruído Gaussiano\nz~N(0,I)"] --> C
        B --> C["FLUX DiT\n(12B params, Flow Matching)"]
        C -->|"ODE: 20 passos"| D["Latente Limpo z₁"]
        D --> E["Decoder VAE\n(83M params)"]
        E --> F["Imagem de Saída\n1024×1024px"]
    ```
