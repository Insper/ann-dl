<div id="quiz-ann"></div>
<script>
buildQuiz('ann', 'Redes Neurais — História', [
  {
    q: 'Quem propôs o primeiro modelo matemático de um neurônio artificial em 1943?',
    opts: ['Rosenblatt e Minsky', 'McCulloch e Pitts', 'Hopfield e Hinton', 'LeCun e Bengio'],
    ans: 1,
    exp: 'Warren McCulloch e Walter Pitts propuseram o neurônio MP em 1943 — um modelo binário que dispara quando a soma ponderada das entradas excede um limiar. Foi a fundação das redes neurais artificiais.'
  },
  {
    q: 'O que foi o "Inverno da IA"?',
    opts: [
      'Clima frio que danificou data centers',
      'Períodos de cortes de financiamento e pessimismo após expectativas excessivas não serem atendidas',
      'A fase em que algoritmos genéticos substituíram redes neurais',
      'O período em que apenas computadores quânticos avançaram'
    ],
    ans: 1,
    exp: 'Dois períodos (1974–80 e 1987–93) de progresso estagnado — motivados pela crítica de Minsky/Papert ao Perceptron, falta de poder computacional e promessas não cumpridas — causaram severa retração no financiamento e interesse em pesquisa de IA.'
  },
  {
    q: 'Qual foi o impacto do AlexNet (2012) na história do aprendizado profundo?',
    opts: [
      'Foi o primeiro modelo a usar gradiente descendente',
      'Venceu o ImageNet por grande margem usando CNNs + GPUs, provando a eficácia do DL em escala',
      'Introduziu o mecanismo de atenção que levou aos Transformers',
      'Criou o primeiro conjunto de dados de imagens em larga escala'
    ],
    ans: 1,
    exp: 'AlexNet (Krizhevsky, Sutskever, Hinton) reduziu o erro top-5 do ImageNet de ~26% para ~15% — uma margem enorme. Mostrou que CNNs profundas treinadas em GPU superavam decisivamente métodos clássicos de visão computacional.'
  },
  {
    q: 'O que é o Problema do Gradiente Desvanecente?',
    opts: [
      'Quando os gradientes ficam tão grandes que os pesos explodem',
      'Quando os gradientes ficam tão pequenos durante a retropropagação que as camadas iniciais aprendem extremamente devagar',
      'Quando a taxa de aprendizado é muito alta',
      'Quando a função de perda oscila sem convergir'
    ],
    ans: 1,
    exp: 'Durante a retropropagação, os gradientes são multiplicados pelas derivadas das funções de ativação (ex: sigmoid: ≤0,25). Em redes profundas, essa cadeia de valores pequenos se aproxima de zero, impedindo que as camadas iniciais aprendam representações significativas.'
  },
  {
    q: 'Quais três pesquisadores são frequentemente chamados de "Padrinhos do Aprendizado Profundo"?',
    opts: [
      'Turing, Shannon e von Neumann',
      'Minsky, McCarthy e Simon',
      'LeCun, Bengio e Hinton (Prêmio Turing 2018)',
      'Goodfellow, Schmidhuber e Hochreiter'
    ],
    ans: 2,
    exp: 'Yann LeCun (CNNs), Yoshua Bengio (modelos de linguagem neurais, RNNs) e Geoffrey Hinton (máquinas de Boltzmann, retropropagação) ganharam o Prêmio Turing de 2018 por contribuições fundamentais ao aprendizado profundo.'
  }
]);
</script>
