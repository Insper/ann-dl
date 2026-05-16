<div id="quiz-llms"></div>
<script>
buildQuiz('llms', 'Grandes Modelos de Linguagem', [
  {
    q: 'Qual é o objetivo de pré-treinamento dos modelos estilo GPT?',
    opts: [
      'Prever a classe de um documento de texto',
      'Prever o próximo token dada a sequência anterior (predição do próximo token)',
      'Comparar pares de frases para determinar implicação',
      'Responder questões de múltipla escolha'
    ],
    ans: 1,
    exp: 'GPT (apenas decoder) é treinado com modelagem de linguagem causal: p(x_t | x_1,...,x_{t-1}). Autossupervisionado — os rótulos são os próprios tokens. Escala para trilhões de tokens sem anotação manual.'
  },
  {
    q: 'O que são capacidades emergentes em LLMs?',
    opts: [
      'Capacidades explicitamente programadas durante o fine-tuning',
      'Capacidades que aparecem abruptamente em modelos acima de certo limiar de escala, ausentes em modelos menores',
      'A velocidade de geração que emerge com otimizações de hardware',
      'A capacidade de um modelo reduzir seu próprio tamanho'
    ],
    ans: 1,
    exp: 'Wei et al. (2022) documentaram que capacidades como aritmética de múltiplos dígitos, raciocínio chain-of-thought e tradução emergem de forma não-linear ao cruzar limiares de escala. Não podem ser explicadas por interpolação simples das capacidades de modelos menores.'
  },
  {
    q: 'O que é RLHF (Reinforcement Learning from Human Feedback)?',
    opts: [
      'Treinar um agente RL para jogar jogos usando feedback humano como recompensa',
      'Alinhar LLMs às preferências humanas usando um modelo de recompensa treinado com rankings humanos',
      'Uma técnica de aumento de dados usando feedback real de usuários',
      'Treinar o tokenizer usando preferências de linguistas'
    ],
    ans: 1,
    exp: 'RLHF (InstructGPT, 2022): 1) SFT em demonstrações humanas; 2) Reward Model aprendendo a ranquear respostas por preferência humana; 3) PPO/DPO para otimizar o LLM contra o RM. Transforma um preditor de texto em um assistente útil.'
  },
  {
    q: 'O que é Mixture of Experts (MoE) em LLMs modernos?',
    opts: [
      'Um ensemble de múltiplos LLMs diferentes votando na resposta final',
      'Substituir cada FFN por E redes especializadas independentes, com um roteador que ativa apenas k por token',
      'Fine-tuning de diferentes partes do modelo por equipes de especialistas diferentes',
      'Um método de quantização usando especialistas para diferentes faixas de valores de pesos'
    ],
    ans: 1,
    exp: 'MoE (ex: Mixtral 8×7B, DeepSeek-V3 671B/37B ativos): cada camada FFN tem E especialistas e um roteador que seleciona os top-k por token. Mais capacidade total com o mesmo custo computacional por forward pass.'
  },
  {
    q: 'O que é prompting Chain-of-Thought (CoT)?',
    opts: [
      'Encadear múltiplos LLMs em série para resolver tarefas complexas',
      'Incluir passos intermediários de raciocínio no prompt para guiar o modelo a resolver problemas passo a passo',
      'Uma técnica de tokenização que vincula palavras semanticamente relacionadas',
      'Treinar o modelo com exemplos de raciocínio lógico formal'
    ],
    ans: 1,
    exp: 'CoT (Wei et al., 2022): adicionar "Vamos pensar passo a passo" ou exemplos com raciocínio explícito melhora dramaticamente matemática, lógica e programação. O modelo usa tokens intermediários como memória de trabalho.'
  }
]);
</script>
