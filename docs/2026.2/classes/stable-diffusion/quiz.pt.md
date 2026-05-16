<div id="quiz-stable-diffusion"></div>
<script>
buildQuiz('stable-diffusion', 'Stable Diffusion & Modelos de Difusão', [
  {
    q: 'O que acontece no processo de difusão direta (forward)?',
    opts: [
      'O modelo progressivamente gera uma imagem a partir do ruído',
      'Ruído Gaussiano é progressivamente adicionado a uma imagem real até ela se tornar ruído puro',
      'A imagem é comprimida pelo encoder do VAE para o espaço latente',
      'A U-Net prevê pesos de atenção para cada timestep'
    ],
    ans: 1,
    exp: 'Forward (adição de ruído): q(x_t|x_{t-1}) = N(√α_t·x_{t-1}, β_t·I). Em T passos, x_T ≈ N(0,I). Este processo é fixo, não treinado. O modelo aprende o processo reverso (denoising). No SD, isso ocorre no espaço latente do VAE.'
  },
  {
    q: 'O que a U-Net prevê no DDPM (Denoising Diffusion Probabilistic Models)?',
    opts: [
      'A imagem limpa original x_0 diretamente',
      'O ruído ε que foi adicionado a cada passo, que é então subtraído',
      'Pesos de atenção entre patches da imagem ruidosa',
      'A probabilidade de cada pixel pertencer à imagem original'
    ],
    ans: 1,
    exp: 'Predição de ε: ε_θ(x_t, t) ≈ ε. A perda é ||ε - ε_θ(x_t, t)||². Conhecer o ruído permite estimar x_0 e calcular x_{t-1}. Alternativa: predição de v (SD 2.x e 3) que é mais estável em timesteps extremos.'
  },
  {
    q: 'O que é o espaço latente no Stable Diffusion (Latent Diffusion Model)?',
    opts: [
      'O espaço de vocabulário do encoder de texto CLIP',
      'Uma representação comprimida de imagens gerada pelo VAE, onde a difusão ocorre em vez do espaço de pixels',
      'O espaço de parâmetros da U-Net durante o treinamento',
      'A distribuição de ruído Gaussiano no timestep T'
    ],
    ans: 1,
    exp: 'LDM (Rombach et al., 2022): difundir diretamente em pixels 512×512×3 é caro. O VAE comprime para 64×64×4 (fator 8×). A difusão ocorre no espaço latente — muito mais eficiente com qualidade comparável.'
  },
  {
    q: 'O que é Classifier-Free Guidance (CFG)?',
    opts: [
      'Usar um classificador separado para guiar a difusão em direção à classe desejada',
      'Treinar o modelo com e sem condicionamento e interpolar predições para amplificar o alinhamento com o prompt',
      'Uma técnica de regularização que remove aleatoriamente o condicionamento',
      'Um método de fine-tuning que não requer rótulos de classe'
    ],
    ans: 1,
    exp: 'CFG: ε_guided = ε_uncond + escala × (ε_cond - ε_uncond). Com escala=7,5, o modelo é fortemente direcionado para o prompt. Durante o treinamento, 50% do tempo o condicionamento é zerado para ensinar predição incondicional.'
  },
  {
    q: 'Qual componente arquitetural permite que a U-Net integre informações do prompt textual?',
    opts: [
      'Batch Normalization nos blocos residuais',
      'Cross-Attention entre features da U-Net e embeddings de texto',
      'Concatenar o embedding de texto com a entrada de ruído',
      'Uma camada softmax final que seleciona tokens de vocabulário'
    ],
    ans: 1,
    exp: 'Nos blocos residuais da U-Net: Q vem das features da imagem ruidosa, K e V vêm dos embeddings do encoder de texto (CLIP/T5). O cross-attention injeta condicionamento textual em cada nível de resolução da U-Net.'
  }
]);
</script>
