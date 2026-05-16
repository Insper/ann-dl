Inteligência Artificial (IA) é um campo amplo que abrange diversas abordagens e técnicas para criar sistemas inteligentes capazes de realizar tarefas que tipicamente requerem inteligência humana. Essas tarefas incluem raciocínio, aprendizado, percepção e tomada de decisão.

A IA pode ser categorizada em três paradigmas principais, cada um com seus pontos fortes e fracos: IA Simbólica, IA Conexionista e IA Neuro-Simbólica. Cada um desses paradigmas tem seus próprios pontos fortes e fracos e são frequentemente usados em diferentes contextos dependendo do problema a ser resolvido.

## Paradigmas de IA
| Paradigma          | Descrição                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| IA Simbólica       | Foca no raciocínio de alto nível e representação do conhecimento usando símbolos e regras. Destaca-se em tarefas que requerem raciocínio lógico, como prova de teoremas e sistemas especialistas. No entanto, tem dificuldades com percepção e aprendizado a partir de dados brutos. Exemplos: sistemas baseados em lógica, sistemas especialistas e grafos de conhecimento. |
| IA Conexionista  | Baseada em redes neurais artificiais (ANNs), destaca-se em reconhecimento de padrões, aprendizado a partir de grandes conjuntos de dados e tratamento de dados ruidosos. É particularmente eficaz em tarefas como reconhecimento de imagem e fala. No entanto, frequentemente carece de interpretabilidade e tem dificuldades com tarefas de raciocínio. Exemplos incluem redes neurais convolucionais (CNNs), redes neurais recorrentes (RNNs) e transformers. |
| IA Neuro-Simbólica | Combina os pontos fortes da IA simbólica e conexionista, visando criar sistemas capazes de raciocinar sobre problemas complexos e ao mesmo tempo aprender a partir dos dados. Aproveita capacidades de raciocínio simbólico junto com redes neurais para aumentar interpretabilidade e habilidades de raciocínio. Exemplos incluem sistemas neuro-simbólicos que integram lógica simbólica com redes neurais, como modelos de linguagem aumentados por conhecimento e redes neurais em grafos. |

```python exec="on" html="1"
--8<-- "docs/2026.2/classes/concepts/relations.py"
```

A IA Neuro-Simbólica combina raciocínio simbólico com redes neurais, aproveitando os pontos fortes de ambas as abordagens. Ela visa criar sistemas capazes de raciocinar sobre problemas complexos e ao mesmo tempo aprender a partir dos dados.

Essa abordagem é particularmente útil em tarefas que requerem tanto raciocínio de alto nível quanto a capacidade de aprender a partir de dados brutos, como compreensão de linguagem natural e tomada de decisão complexa.

Existem várias abordagens para implementar IA. O aprendizado de máquina (AM) é um dos métodos mais comuns, onde algoritmos aprendem a partir de dados para fazer previsões ou decisões. Redes neurais, um subconjunto do AM, são inspiradas pela estrutura e função do cérebro humano e são particularmente eficazes em tarefas como reconhecimento de imagem e fala. O aprendizado profundo, uma forma mais avançada de redes neurais, usa múltiplas camadas de processamento para extrair padrões complexos de grandes conjuntos de dados.

```python exec="on" html="1"
--8<-- "docs/2026.2/classes/concepts/hierarchical.py"
```

## Aprendizado de Máquina

No contexto da IA, técnicas de aprendizado de máquina (AM) são usadas para permitir que sistemas aprendam a partir de dados e melhorem seu desempenho ao longo do tempo sem serem explicitamente programados. Essas técnicas permitem que sistemas de IA se adaptem e generalizem a partir de exemplos, tornando-os capazes de lidar com uma ampla variedade de tarefas, desde reconhecimento de imagem até processamento de linguagem natural.

As técnicas são frequentemente divididas em duas categorias principais: **aprendizado supervisionado** e **aprendizado não supervisionado**.

!!! info "Aprendizado Supervisionado"
    
    O **aprendizado supervisionado** envolve treinar um modelo em dados rotulados, onde os dados de entrada são pareados com a saída correta. Isso permite que o modelo aprenda padrões e faça previsões baseadas em novos dados não vistos.
    
    Essa abordagem é particularmente eficaz quando há uma relação clara entre as features de entrada e os rótulos de saída, permitindo que o modelo generalize a partir dos dados de treinamento para fazer previsões precisas em novos dados. Exemplos incluem tarefas de classificação (ex: identificar objetos em imagens) e tarefas de regressão (ex: prever preços de casas com base em features).

!!! info "Aprendizado Não Supervisionado"
    
    O **aprendizado não supervisionado**, por outro lado, envolve treinar um modelo em dados não rotulados, onde o modelo deve encontrar padrões e relacionamentos nos dados sem orientação explícita.
    
    Essa abordagem é útil para descobrir estruturas ocultas nos dados, como clusters ou grupos, sem conhecimento prévio dos rótulos. É frequentemente usada em análise exploratória de dados e extração de features. Exemplos incluem tarefas de clusterização (ex: agrupar documentos semelhantes) e tarefas de redução de dimensionalidade (ex: reduzir o número de features em um conjunto de dados preservando informações importantes).

Existem também técnicas de **aprendizado semi-supervisionado**, que combinam dados rotulados e não rotulados para melhorar o desempenho do modelo. Essa abordagem é particularmente útil quando dados rotulados são escassos ou caros de obter, permitindo que o modelo aproveite a abundância de dados não rotulados para aprimorar seu aprendizado.

Além disso, existem técnicas de **aprendizado por reforço**, onde um agente aprende a tomar decisões interagindo com um ambiente e recebendo feedback na forma de recompensas ou penalidades. Essa abordagem é particularmente eficaz para tarefas que envolvem tomada de decisão sequencial, como jogar games ou controle robótico.

Técnicas de aprendizado de máquina abordam uma ampla variedade de problemas, principalmente através de **classificação** e **regressão**, que são tarefas centrais do aprendizado supervisionado. A classificação envolve prever rótulos ou categorias discretas com base em features de entrada, enquanto a regressão foca em prever valores contínuos. Essas abordagens são extensivamente aplicadas em domínios como reconhecimento de imagem, processamento de linguagem natural e previsão de séries temporais.

Alguns exemplos de técnicas de aprendizado de máquina:

| Técnica          | Descrição                                                                                   |
|---|---|
| Árvores de Decisão    | Um modelo em forma de árvore usado para tarefas de classificação e regressão, onde cada nó interno representa uma feature, cada ramo representa uma regra de decisão e cada nó folha representa um resultado.
| Random Forest     | Um método ensemble que combina múltiplas árvores de decisão para melhorar a precisão e reduzir o overfitting. Funciona treinando múltiplas árvores de decisão em diferentes subconjuntos dos dados e calculando a média de suas previsões.
| Support Vector Machines (SVM) | Um algoritmo de aprendizado supervisionado que encontra o hiperplano ótimo para separar classes diferentes no espaço de features. É eficaz para dados de alta dimensionalidade e pode lidar com tarefas de classificação linear e não-linear.
| K-Nearest Neighbors (KNN) | Um algoritmo simples que classifica novas instâncias com base na classe majoritária de seus k vizinhos mais próximos no espaço de features. É um método não-paramétrico que pode ser usado para tarefas de classificação e regressão. |
| Naive Bayes       | Um classificador probabilístico baseado no teorema de Bayes, assumindo independência entre features. É particularmente eficaz para tarefas de classificação de texto, como detecção de spam e análise de sentimentos. |
| Regressão Linear | Um método estatístico usado para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes ajustando uma equação linear aos dados observados. É comumente usado para prever resultados contínuos com base em features de entrada. |
| Regressão Logística | Um método estatístico usado para tarefas de classificação binária, onde a saída é uma probabilidade que pode ser mapeada para duas classes. Modela a relação entre features de entrada e o log-odds do resultado usando uma função logística. |
| K-Means Clustering | Um algoritmo de aprendizado não supervisionado que particiona dados em k clusters com base na similaridade de features. Itera atribuindo pontos de dados ao centroide do cluster mais próximo e atualiza os centroides até convergir. |
| Análise de Componentes Principais (PCA) | Uma técnica de redução de dimensionalidade que transforma dados de alta dimensionalidade em um espaço de menor dimensão preservando features importantes. Identifica os componentes principais que capturam a maior variância nos dados, sendo útil para visualização e extração de features. |
| Gradient Boosting | Uma técnica de aprendizado ensemble que constrói uma série de aprendizes fracos (geralmente árvores de decisão) de forma sequencial, onde cada novo aprendiz corrige os erros dos anteriores. É eficaz tanto para tarefas de classificação quanto de regressão e é amplamente usado em competições de aprendizado de máquina. |

## Redes Neurais

Redes neurais são uma classe de modelos de aprendizado de máquina inspirados na estrutura e função do cérebro humano. Consistem em nós interconectados (neurônios) organizados em camadas, onde cada conexão tem um peso associado que é ajustado durante o treinamento. Redes neurais são particularmente eficazes para tarefas que envolvem padrões complexos, como reconhecimento de imagem e fala.

As redes neurais podem ser categorizadas em vários tipos, incluindo:

- **Redes Neurais Feedforward (FNNs)**: O tipo mais simples de rede neural, onde a informação flui em uma direção, da entrada para a saída, sem ciclos. São comumente usadas para tarefas como classificação e regressão.
- **Redes Neurais Convolucionais (CNNs)**: Redes neurais especializadas projetadas para processar dados em grade, como imagens. Usam camadas convolucionais para aprender automaticamente hierarquias espaciais de features, tornando-as altamente eficazes para tarefas de reconhecimento de imagem.
- **Redes Neurais Recorrentes (RNNs)**: Redes neurais projetadas para dados sequenciais, como séries temporais ou linguagem natural. Têm conexões que retornam sobre si mesmas, permitindo manter uma memória de entradas anteriores. Isso as torna adequadas para tarefas como modelagem de linguagem e reconhecimento de fala.
- **Transformers**: Um tipo de arquitetura de rede neural que usa mecanismos de self-attention para processar sequências de dados. Revolucionaram tarefas de processamento de linguagem natural, permitindo que modelos como BERT e GPT alcancem desempenho de estado da arte em várias tarefas de compreensão de linguagem.

## Aprendizado Profundo

O aprendizado profundo (deep learning) é um subconjunto do aprendizado de máquina que foca em usar redes neurais profundas com muitas camadas para aprender representações complexas dos dados. Alcançou sucesso notável em vários domínios, incluindo visão computacional, processamento de linguagem natural e reconhecimento de fala. Modelos de aprendizado profundo são capazes de aprender automaticamente features hierárquicas a partir de dados brutos, eliminando a necessidade de engenharia manual de features. Isso levou a avanços significativos em aplicações de IA, permitindo que sistemas realizem tarefas anteriormente consideradas desafiadoras ou impossíveis.


## Recursos Adicionais

<iframe width="100%" height="470" src="https://www.youtube.com/embed/21EiKfQYZXc" allowfullscreen></iframe>


[^1]: [Wiki - IA Neuro-Simbólica](https://en.wikipedia.org/wiki/Neuro-symbolic_AI){target='_blank'}
[^2]: 2020, Forbes - [Symbolism Versus Connectionism In AI: Is There A Third Way?](https://www.forbes.com/councils/forbestechcouncil/2020/09/01/symbolism-versus-connectionism-in-ai-is-there-a-third-way/){target='_blank'}
[^3]: Garcez, A.d., Lamb, L.C. Neurosymbolic AI: the 3rd wave. Artif Intell Rev 56, 12387–12406 (2023). [doi.org/10.1007/s10462-023-10448-w](https://doi.org/10.1007/s10462-023-10448-w){target='_blank'} [:octicons-download-24:](https://arxiv.org/abs/2012.05876){target='_blank'}



---

--8<-- "docs/2026.2/classes/concepts/quiz.pt.md"
