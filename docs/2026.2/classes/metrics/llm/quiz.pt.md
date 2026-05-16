<div id="quiz-metrics-llm"></div>
<script>
buildQuiz('metrics-llm', 'Métricas de LLM', [
  {
    q: 'O que a Perplexidade mede em modelos de linguagem?',
    opts: [
      'A acurácia do modelo em tarefas de classificação',
      'O quanto o modelo se surpreende com os tokens do corpus de teste — PPL menor indica melhor ajuste',
      'O número de parâmetros necessários para gerar texto coerente',
      'A velocidade de inferência em tokens por segundo'
    ],
    ans: 1,
    exp: 'PPL = exp(-1/N · Σlog p(wᵢ|w<i)). PPL=10 significa que o modelo está, em média, tão incerto quanto entre 10 escolhas igualmente prováveis. Baixa perplexidade ≠ modelo útil ou alinhado.'
  },
  {
    q: 'O que o benchmark MMLU avalia?',
    opts: [
      'Geração de código em múltiplas linguagens de programação',
      'Conhecimento multidisciplinar em 57 disciplinas via questões de múltipla escolha',
      'Capacidade de seguir instruções em diálogos de múltiplos turnos',
      'Raciocínio matemático avançado em problemas de cálculo'
    ],
    ans: 1,
    exp: 'MMLU (Massive Multitask Language Understanding) abrange 57 disciplinas — medicina, direito, história, física, etc. É o padrão ouro para medir a amplitude do conhecimento adquirido durante o pré-treinamento.'
  },
  {
    q: 'Por que o Chatbot Arena (Classificação ELO) é considerado uma das métricas mais confiáveis para LLMs?',
    opts: [
      'Porque usa um algoritmo matemático rigoroso sem intervenção humana',
      'Porque reflete preferências humanas reais via comparações A/B, capturando aspectos que benchmarks automáticos perdem',
      'Porque é mais rápido de calcular que outros benchmarks',
      'Porque penaliza diretamente alucinações na métrica'
    ],
    ans: 1,
    exp: 'No Chatbot Arena, usuários reais comparam duas respostas de forma cega (sem saber qual modelo) e votam na melhor. A classificação ELO captura utilidade real, estilo de resposta e criatividade — aspectos impossíveis de medir automaticamente.'
  },
  {
    q: 'O que é Faithfulness em sistemas RAG?',
    opts: [
      'A similaridade semântica entre pergunta e resposta',
      'A proporção de afirmações na resposta que são suportadas pelos documentos recuperados',
      'A taxa de acerto do retriever na busca de documentos relevantes',
      'A latência de resposta do sistema RAG'
    ],
    ans: 1,
    exp: 'Faithfulness mede se o LLM "inventou" informações ou se cada afirmação na resposta pode ser atribuída ao contexto recuperado. Alta faithfulness indica respostas fundamentadas, não alucinações.'
  },
  {
    q: 'Qual é o principal viés conhecido na avaliação LLM-as-Judge?',
    opts: [
      'Modelos juízes sempre preferem respostas em inglês',
      'Tendência a preferir respostas mais longas e verbosas, independentemente da qualidade real',
      'Incapacidade de avaliar respostas que contêm código',
      'Sempre necessitar de uma resposta de referência humana'
    ],
    ans: 1,
    exp: 'Avaliadores LLM tendem a favorecer respostas longas e bem formatadas e linguagem similar à sua. Outros vieses incluem viés de posição (favorecer a primeira resposta mostrada) e auto-preferência (favorecer saídas da mesma família de modelos).'
  }
]);
</script>
