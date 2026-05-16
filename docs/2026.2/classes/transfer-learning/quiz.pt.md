<div id="quiz-transfer-learning"></div>
<script>
buildQuiz('transfer-learning', 'Transfer Learning & Fine-Tuning', [
  {
    q: 'No Transfer Learning, o que significa "congelar" (freezing) camadas?',
    opts: [
      'Salvar os pesos em disco para uso posterior',
      'Impedir que os pesos sejam atualizados durante o treinamento, preservando as representações pré-treinadas',
      'Inicializar os pesos com temperatura baixa (próxima de zero)',
      'Aplicar 100% de dropout nessas camadas'
    ],
    ans: 1,
    exp: 'Congelar define requires_grad=False para os parâmetros. Reduz memória e evita destruir representações pré-treinadas úteis. As camadas iniciais (detectores de bordas, padrões de textura) raramente precisam ser atualizadas para novas tarefas.'
  },
  {
    q: 'O que o LoRA (Low-Rank Adaptation) faz?',
    opts: [
      'Remove camadas do modelo para reduzir memória',
      'Adiciona matrizes treináveis de baixo rank (ΔW = BA) em paralelo aos pesos congelados',
      'Treina apenas o bias de cada camada, congelando os pesos',
      'Quantiza os pesos para 4 bits para reduzir memória'
    ],
    ans: 1,
    exp: 'LoRA congela W₀ e adiciona W = W₀ + BA onde B∈R^{d×r}, A∈R^{r×k} com r≪min(d,k). Para um modelo de 7B parâmetros com r=8, apenas ~0,1% dos parâmetros são treináveis, com qualidade comparável ao fine-tuning completo.'
  },
  {
    q: 'Qual é a diferença entre Feature Extraction e Full Fine-Tuning?',
    opts: [
      'Feature Extraction treina todas as camadas; Full Fine-Tuning congela tudo',
      'Feature Extraction congela o backbone e treina apenas o head; Full Fine-Tuning atualiza todos os pesos',
      'São equivalentes com taxas de aprendizado diferentes',
      'Feature Extraction usa gradiente descendente; Full Fine-Tuning usa métodos evolutivos'
    ],
    ans: 1,
    exp: 'Feature Extraction: backbone congelado → features extraídas → novo classificador treinado do zero. Rápido, poucos dados necessários. Full FT: todos os pesos ajustados com lr pequena. Melhor qualidade, mais dados e GPU necessários.'
  },
  {
    q: 'O que é QLoRA?',
    opts: [
      'LoRA aplicado exclusivamente à camada de atenção Q (query)',
      'LoRA combinado com quantização em 4 bits do modelo base, reduzindo drasticamente o uso de VRAM',
      'Uma versão quantizada do otimizador Adam para treinamento LoRA mais rápido',
      'Q-Learning combinado com LoRA para fine-tuning por reforço'
    ],
    ans: 1,
    exp: 'QLoRA (Dettmers et al., 2023) quantiza o modelo base para NF4 (4 bits) — reduzindo memória ~4×. Os adaptadores LoRA permanecem em fp16. Permite fine-tuning do LLaMA-2 70B em uma única GPU de 48GB.'
  },
  {
    q: 'Quando a Adaptação de Domínio é necessária antes do fine-tuning de tarefa?',
    opts: [
      'Nunca — o fine-tuning direto de tarefa é sempre suficiente',
      'Quando o domínio alvo (médico, jurídico, código) tem vocabulário e estilo muito diferentes dos dados de pré-treinamento',
      'Quando o modelo tem mais de 1B de parâmetros',
      'Quando os dados de fine-tuning estão desbalanceados'
    ],
    ans: 1,
    exp: 'A Adaptação de Domínio (pré-treinamento continuado) em texto não rotulado do domínio ajuda o modelo a aprender terminologia e estilo específicos do domínio. Exemplo: BioGPT faz adaptação de domínio em literatura médica antes do fine-tuning em NER clínico.'
  }
]);
</script>
