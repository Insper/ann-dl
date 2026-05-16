<div id="quiz-concepts"></div>
<script>
buildQuiz('concepts', 'Conceitos de IA', [
  {
    q: 'O que melhor diferencia Deep Learning do Machine Learning clássico?',
    opts: [
      'DL usa redes neurais profundas para aprender representações hierárquicas automaticamente a partir de dados brutos',
      'DL não requer dados de treinamento',
      'DL é exclusivo para tarefas de visão computacional',
      'DL usa apenas regras simbólicas definidas por especialistas do domínio'
    ],
    ans: 0,
    exp: 'Deep Learning usa múltiplas camadas de processamento para aprender representações progressivamente mais abstratas, eliminando a necessidade de engenharia manual de features. O ML clássico tipicamente requer features artesanais.'
  },
  {
    q: 'Qual paradigma de IA combina raciocínio simbólico com redes neurais?',
    opts: ['IA Conexionista', 'IA Simbólica', 'IA Neuro-Simbólica', 'Aprendizado por Reforço'],
    ans: 2,
    exp: 'A IA Neuro-Simbólica combina os pontos fortes do raciocínio lógico simbólico com a capacidade de aprendizado das redes neurais, como em Modelos de Linguagem Aumentados por Conhecimento e Redes Neurais em Grafos.'
  },
  {
    q: 'No aprendizado supervisionado, o que o modelo recebe durante o treinamento?',
    opts: [
      'Apenas dados não rotulados',
      'Pares entrada-saída com rótulos corretos',
      'Recompensas do ambiente',
      'Regras lógicas codificadas manualmente'
    ],
    ans: 1,
    exp: 'No aprendizado supervisionado, cada amostra de treinamento tem um rótulo correto associado. O modelo aprende a mapear entradas para saídas minimizando o erro de predição. Exemplos: classificação, regressão.'
  },
  {
    q: 'Qual técnica de ML é mais adequada para treinar um agente para jogar um videogame?',
    opts: ['Regressão Linear', 'Clusterização K-Means', 'Aprendizado por Reforço', 'PCA'],
    ans: 2,
    exp: 'O Aprendizado por Reforço treina agentes que interagem com um ambiente, recebendo recompensas ou penalidades, aprendendo políticas ótimas de ação — ideal para tarefas de decisão sequencial como jogos.'
  },
  {
    q: 'O que é o Teste de Turing?',
    opts: [
      'Um benchmark para medir a perplexidade de LLMs',
      'Um critério para avaliar se uma máquina exibe comportamento inteligente indistinguível de um humano',
      'Uma métrica de acurácia para classificadores',
      'Um método de regularização para redes neurais'
    ],
    ans: 1,
    exp: 'Proposto por Alan Turing em 1950, o teste avalia se um juiz humano consegue distinguir respostas de máquina de respostas humanas em uma conversa textual. Se o juiz não conseguir, a máquina passa no teste.'
  }
]);
</script>
