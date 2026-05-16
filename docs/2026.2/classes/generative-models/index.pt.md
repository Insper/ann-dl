
IA Generativa, frequentemente abreviada como GenAI, é um subcampo da inteligência artificial que emprega modelos generativos para criar novos conteúdos — como texto, imagens, vídeos, áudio, código de software ou outras formas de dados — aprendendo padrões e estruturas a partir de vastos datasets de treinamento. Esses modelos tipicamente respondem a prompts do usuário (ex: entradas em linguagem natural) produzindo saídas originais que imitam o estilo ou as características dos dados aprendidos, distinguindo-os de sistemas de IA tradicionais que principalmente analisam ou preveem informações existentes. Exemplos comuns incluem ferramentas como ChatGPT para geração de texto e DALL-E para criação de imagens, impulsionadas por técnicas como grandes modelos de linguagem (LLMs) ou redes adversariais generativas (GANs).

As raízes da IA Generativa remontam a modelos de probabilidade iniciais, evoluindo por sistemas baseados em regras até o aprendizado profundo. Apresentamos cronologicamente, com destaques detalhados sobre eventos transformadores.


| Era/Ano | Desenvolvimento Chave | Detalhes/Destaques | Impacto |
|---------|-----------------|--------------------|--------|
| Anos 1950: Fundações Probabilísticas | Cadeias de Markov ([1953, Claude Shannon](https://www.cs.princeton.edu/courses/archive/fall13/cos126/assignments/markov.html){:target="_blank"}) | O trabalho de Shannon em teoria da informação introduziu modelos de Markov para geração de texto. Destaque: Primeiro "poema de IA" gerado via cadeias — rudimentar, mas provou que máquinas poderiam imitar padrões. | Estabeleceu base para geração de sequências; influenciou PLN. |
| Anos 1980: Redes Neurais Iniciais | Máquinas de Boltzmann ([1986, Geoffrey Hinton et al.](https://doi.org/10.1016/S0364-0213(85)80012-4){:target="_blank"}) | RBMs usadas em modelos baseados em energia para aprender distribuições de dados. Destaque: algoritmo "wake-sleep" de Hinton (1995) treinou redes não supervisionadas em imagens — primeiros vislumbres de "sonhar" generativo. | Ponte para o aprendizado profundo; usado em sistemas de recomendação iniciais. |
| Anos 1990: Inferência Variacional | Precursores dos Autoencoders Variacionais (VAEs) | Métodos bayesianos para modelos de variáveis latentes. Destaque: EM variacional de Jordan & Weiss (1998) — artigo chave permitindo aproximações posteriores tratáveis. | Possibilitou modelagem generativa escalável; base para VAEs modernos. |
| Anos 2010: Boom do Aprendizado Profundo | Deep Belief Nets (2006, Hinton) → VAEs (2013, Kingma & Welling) | RBMs empilhadas pré-treinando redes profundas. Destaque: artigo dos VAEs (ICLR 2014) introduziu o truque de reparametrização para retropropagação através de nós estocásticos — gerou dígitos MNIST borrados, mas escalável. | Democratizou o aprendizado não supervisionado; VAEs na descoberta de medicamentos. |
| 2014-Presente: Era Adversarial | GANs ([2014, Goodfellow et al.](https://arxiv.org/abs/1406.2661){:target="_blank"}) | Posterior: [StyleGAN (2018, NVIDIA)](https://github.com/NVlabs/stylegan){:target="_blank"} para rostos fotorrealistas. Destaque: estreia das GANs no NIPS 2014 gerou quartos realistas — chocou a comunidade, iniciando o paradigma de "treinamento adversarial". | Explodiu as aplicações (ex: DeepFakes 2017); debates éticos. |
| Anos 2020: Escala e Multimodalidade | Modelos de Difusão (2020, Ho et al.); GPT-3 (2020, OpenAI) | DDPMs (Denoising Diffusion Probabilistic Models). Destaque: [DALL·E (2021)](https://venturebeat.com/ai/openai-debuts-dall-e-for-generating-images-from-text){:target="_blank"} combinou difusão + transformers para texto-para-imagem; Stable Diffusion (2022) open-source, gerando mais de 1 bilhão de imagens/mês. | IA generativa multimodal (ex: geração de vídeo Sora, 2024); preocupações com uso de energia. |


[^1]: Geeks for Geeks - [What is Generative AI?](https://www.geeksforgeeks.org/artificial-intelligence/what-is-generative-ai/){:target="_blank"}
[^2]: [A Brief History of Generative Models](https://medium.com/@jimcanary/a-brief-history-and-overview-of-generative-models-in-machine-learning-8881ee7ff87b){:target="_blank"}


---

--8<-- "docs/2026.2/classes/generative-models/quiz.pt.md"
