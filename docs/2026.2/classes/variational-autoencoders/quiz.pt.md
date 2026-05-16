<div id="quiz-vae"></div>
<script>
buildQuiz('vae', 'Autoencoders Variacionais (VAE)', [
  {
    q: 'O que é o "truque da reparametrização" nos VAEs?',
    opts: [
      'Usar uma função de perda diferente para o encoder e decoder',
      'Amostrar z = μ + σ·ε onde ε~N(0,I), separando o nó estocástico para permitir retropropagação',
      'Substituir o amostrador por uma função determinística',
      'Normalizar os pesos do encoder para norma unitária'
    ],
    ans: 1,
    exp: 'Amostrar diretamente z~N(μ,σ²) não é diferenciável. O truque reescreve z = μ + σ·ε com ε~N(0,I), transferindo a aleatoriedade para um nó externo. Agora ∂z/∂μ=1 e ∂z/∂σ=ε — a retropropagação funciona através de μ e σ.'
  },
  {
    q: 'O que o encoder de um VAE produz, diferentemente de um autoencoder simples?',
    opts: [
      'Um único ponto determinístico z no espaço latente',
      'Os parâmetros μ e σ de uma distribuição Gaussiana q(z|x), não z diretamente',
      'Uma distribuição categórica sobre um vocabulário latente discreto',
      'Um vetor binário selecionando dimensões do espaço latente'
    ],
    ans: 1,
    exp: 'Encoder simples: z = f(x) (determinístico). Encoder VAE: μ(x), σ(x) (estatísticas da distribuição). Isso força o espaço latente a ser contínuo e estruturado, permitindo interpolação e geração de novas amostras.'
  },
  {
    q: 'A perda ELBO de um VAE tem dois termos. O que eles medem?',
    opts: [
      'Perda de reconstrução (MSE) + divergência KL entre q(z|x) e p(z)=N(0,I)',
      'Perda adversarial + perda de reconstrução pixel a pixel',
      'Cross-entropy + perda contrastiva',
      'Perda do encoder + perda do decoder calculadas separadamente'
    ],
    ans: 0,
    exp: 'ELBO = E[log p(x|z)] - KL(q(z|x)||p(z)). O primeiro termo maximiza a qualidade da reconstrução; o segundo regulariza o espaço latente para ser próximo de N(0,I), garantindo estrutura e capacidade de amostrar novas instâncias.'
  },
  {
    q: 'Como gerar novas amostras de um VAE treinado?',
    opts: [
      'Passar uma imagem pelo encoder e reconstruí-la com o decoder',
      'Amostrar z~N(0,I) diretamente e decodificar: x̂ = Decoder(z)',
      'Interpolar entre dois pontos do dataset no espaço de pixels',
      'Usar o encoder invertido para mapear da saída para o espaço latente'
    ],
    ans: 1,
    exp: 'O prior p(z)=N(0,I) é a distribuição de amostragem. A divergência KL durante o treinamento alinha q(z|x) com ele. Na inferência: amostrar z~N(0,I), aplicar o decoder, obter uma nova imagem plausível sem nenhuma entrada.'
  },
  {
    q: 'Por que imagens geradas por VAE tendem a ser mais borradas que saídas de GAN?',
    opts: [
      'Porque VAEs usam menos parâmetros que GANs',
      'Porque a perda de reconstrução (ex: MSE) penaliza todas as frequências igualmente, levando a médias que parecem borradas',
      'Porque o espaço latente do VAE é menor que o de um GAN',
      'Porque VAEs não usam redes convolucionais'
    ],
    ans: 1,
    exp: 'MSE aproxima log-verossimilhança assumindo distribuição Gaussiana pixel a pixel — ignorando estrutura perceptual de alta frequência. GANs aprendem uma perda perceptual implícita via discriminador. Híbridos VAE-GAN ou perdas perceptuais mitigam isso.'
  }
]);
</script>
