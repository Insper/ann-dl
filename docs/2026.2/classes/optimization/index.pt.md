
O Gradiente Descendente é um algoritmo de otimização usado para minimizar a função de perda em modelos de aprendizado de máquina. Ele funciona ajustando iterativamente os parâmetros do modelo na direção do declive mais acentuado da função de perda, definida pelo gradiente negativo.

A ideia principal é atualizar os parâmetros do modelo na direção oposta ao gradiente:

$$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$

/// caption
Gradiente Descendente Puro (Vanilla)
///

onde:

- $\theta$ representa os parâmetros do modelo,
- $\eta$ é a taxa de aprendizado (hiperparâmetro que controla o tamanho do passo),
- $\nabla J(\theta)$ é o gradiente da função de perda em relação aos parâmetros.

Um **Método Baseado em Gradiente** é um algoritmo que encontra os mínimos de uma função, assumindo que se pode calcular facilmente o gradiente dessa função. Assume que a função é contínua e diferenciável quase em todo lugar[^1].

**Intuição do Gradiente Descendente**: Imagine estar em uma montanha em uma noite com neblina. Como você quer descer ao vale e tem visibilidade limitada, olha ao redor para encontrar a direção de descida mais acentuada e dá um passo nessa direção[^1].

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/optimization/gradient-descent-path.py"
```

## Variantes do Gradiente Descendente

Existem várias variantes do gradiente descendente, cada uma com suas características:

1. **Batch Gradient Descent**: Calcula o gradiente usando todo o conjunto de dados. Fornece uma estimativa estável do gradiente, mas pode ser lento para grandes conjuntos de dados:

    $$\theta = \theta - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla J(\theta; x^{(i)}, y^{(i)})$$

    ```python
    for epoch in range(num_epochs):
        gradients = compute_gradients(X, y, model)
        model.parameters -= learning_rate * gradients
    ```

2. **Stochastic Gradient Descent (SGD)**: Calcula o gradiente usando um único ponto de dados por vez. Introduz ruído no processo de otimização, o que pode ajudar a escapar de mínimos locais:

    $$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t; x^{(i)}, y^{(i)})$$

    ```python
    for epoch in range(num_epochs):
        for i in range(len(X)):
            gradients = compute_gradients(X[i], y[i], model)
            model.parameters -= learning_rate * gradients
    ```

3. **Mini-Batch Gradient Descent**: Combina as vantagens do batch e do SGD usando um pequeno subconjunto aleatório dos dados (mini-batch) para calcular o gradiente:

    $$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i=1}^{B} \nabla J(\theta_t; x^{(i)}, y^{(i)})$$

    ```python
    for epoch in range(num_epochs):
        for batch in get_mini_batches(X, y, batch_size):
            gradients = compute_gradients(batch.X, batch.y, model)
            model.parameters -= learning_rate * gradients
    ```


## Momentum

O Momentum é uma variante do gradiente descendente que ajuda a acelerar a otimização usando gradientes passados para suavizar as atualizações. Introduz um termo de "momento" que acumula os gradientes passados:

$$V_{t+1} = \beta V_t + (1 - \beta) \nabla J(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta V_{t+1}$$

$V$ pode ser visto como uma média móvel dos gradientes. Moveremos $\theta$ na direção do novo momento $V$ [^1].

O termo de momentum ajuda a amortecer oscilações e pode levar a uma convergência mais rápida, especialmente em cenários com gradientes ruidosos ou vales na superfície de perda.


## RMSProp

RMSProp (Root Mean Square Propagation) é um algoritmo de otimização com taxa de aprendizado adaptativa que mantém uma média móvel dos gradientes ao quadrado:

$$V_{t+1} = \beta V_t + (1 - \beta) \nabla (\theta_t)^2$$

$$\theta_{t+1} = \theta_t - \eta\frac{\nabla (\theta_t)}{\sqrt{V_{t+1} + \epsilon}}$$

onde:
- $V_t$ é a média móvel dos gradientes ao quadrado,
- $\epsilon$ é uma pequena constante para estabilidade numérica.


## Otimizador ADAM

ADAM (Adaptive Moment Estimation) é um algoritmo de otimização avançado que combina os benefícios do AdaGrad e do RMSProp. Mantém dois momentos para cada parâmetro:

1. Calcular os gradientes: $g_t = \nabla J(\theta_t)$

2. Atualizar o primeiro momento: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$

3. Atualizar o segundo momento: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

4. Estimativas com correção de viés:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. Atualizar os parâmetros:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Os valores padrão dos hiperparâmetros são $\beta_1 = 0.9$, $\beta_2 = 0.999$ e $\epsilon = 10^{-8}$[^2][^3].

---8<-- "docs/2026.2/classes/optimization/comparison.md"

---

## Visualização Interativa: Comparação de Otimizadores

Os otimizadores percorrem diferentes trajetórias em uma superfície de perda com um vale estreito e elíptico — um cenário clássico que desafia SGD mas favorece métodos adaptativos.

<div id="optim-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="optim-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:1rem;">
  <button onclick="optimRun()" id="optim-btn" style="padding:6px 20px;background:#3fb950;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">&#9654; Animar</button>
  <button onclick="optimReset()" style="padding:6px 16px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">&#8635; Resetar</button>
</div>
<div style="display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;margin-top:.8rem;font-size:.85rem;">
  <span><span style="color:#ff7b72;font-weight:bold;">━━</span> SGD</span>
  <span><span style="color:#58a6ff;font-weight:bold;">━━</span> SGD + Momentum</span>
  <span><span style="color:#3fb950;font-weight:bold;">━━</span> Adam</span>
  <span><span style="color:#d29922;font-weight:bold;">━━</span> RMSProp</span>
</div>
</div>

<script>
(function() {
  const canvas = document.getElementById('optim-canvas');
  const ctx = canvas.getContext('2d');
  let animId = null, running = false, step = 0;
  const MAX_STEPS = 80;

  function loss(x,y) { return 0.15*x*x + 2.5*y*y; }
  function grad(x,y) { return [0.3*x, 5*y]; }

  const start = [3.5, 2.8];
  const eta = 0.08;

  const optimizers = {
    sgd: { pos: [...start], color: '#ff7b72', path: [[...start]] },
    momentum: { pos: [...start], v: [0,0], color: '#58a6ff', path: [[...start]] },
    adam: { pos: [...start], m: [0,0], v2: [0,0], t: 0, color: '#3fb950', path: [[...start]] },
    rmsprop: { pos: [...start], v2: [0,0], color: '#d29922', path: [[...start]] },
  };

  function stepOptimizers() {
    const gs = grad(...optimizers.sgd.pos);
    optimizers.sgd.pos[0] -= eta * gs[0];
    optimizers.sgd.pos[1] -= eta * gs[1];
    optimizers.sgd.path.push([...optimizers.sgd.pos]);

    const gm = grad(...optimizers.momentum.pos);
    optimizers.momentum.v[0] = 0.9*optimizers.momentum.v[0] - eta*gm[0];
    optimizers.momentum.v[1] = 0.9*optimizers.momentum.v[1] - eta*gm[1];
    optimizers.momentum.pos[0] += optimizers.momentum.v[0];
    optimizers.momentum.pos[1] += optimizers.momentum.v[1];
    optimizers.momentum.path.push([...optimizers.momentum.pos]);

    const ga = grad(...optimizers.adam.pos);
    optimizers.adam.t++;
    const t = optimizers.adam.t;
    const b1=0.9, b2=0.999, eps=1e-8;
    for (let d=0;d<2;d++) {
      optimizers.adam.m[d] = b1*optimizers.adam.m[d] + (1-b1)*ga[d];
      optimizers.adam.v2[d] = b2*optimizers.adam.v2[d] + (1-b2)*ga[d]*ga[d];
      const mhat = optimizers.adam.m[d]/(1-Math.pow(b1,t));
      const vhat = optimizers.adam.v2[d]/(1-Math.pow(b2,t));
      optimizers.adam.pos[d] -= 0.15 * mhat / (Math.sqrt(vhat)+eps);
    }
    optimizers.adam.path.push([...optimizers.adam.pos]);

    const gr = grad(...optimizers.rmsprop.pos);
    for (let d=0;d<2;d++) {
      optimizers.rmsprop.v2[d] = 0.9*optimizers.rmsprop.v2[d] + 0.1*gr[d]*gr[d];
      optimizers.rmsprop.pos[d] -= eta * gr[d] / (Math.sqrt(optimizers.rmsprop.v2[d])+1e-8);
    }
    optimizers.rmsprop.path.push([...optimizers.rmsprop.pos]);
  }

  function drawFrame() {
    const W = canvas.parentElement.offsetWidth - 48;
    const H = Math.round(W * 0.5);
    canvas.width = W; canvas.height = H;
    canvas.style.height = H + 'px';

    const cx = W/2, cy = H/2;
    const sx = W / 10, sy = H / 7;

    function toCanvas(x, y) { return [cx + x*sx, cy - y*sy]; }

    ctx.fillStyle = '#161b22'; ctx.fillRect(0,0,W,H);

    const levels = [0.1, 0.5, 1.5, 3.5, 7, 12, 18];
    levels.forEach((l, li) => {
      const alpha = 0.15 + li * 0.05;
      ctx.strokeStyle = `rgba(88,166,255,${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const a = Math.sqrt(l/0.15), b = Math.sqrt(l/2.5);
      for (let i=0;i<=100;i++) {
        const angle = i/100*2*Math.PI;
        const [px,py] = toCanvas(a*Math.cos(angle), b*Math.sin(angle));
        i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
      }
      ctx.stroke();
    });

    const [mx0,my0] = toCanvas(0,0);
    ctx.strokeStyle='#ffffff44'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(mx0,my0,8,0,2*Math.PI); ctx.stroke();
    ctx.fillStyle='#ffffff66'; ctx.font='10px monospace'; ctx.textAlign='left';
    ctx.fillText('mín', mx0+10, my0+4);

    Object.values(optimizers).forEach(opt => {
      const path = opt.path;
      if (path.length < 2) return;
      ctx.strokeStyle = opt.color + 'aa'; ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach(([x,y], i) => {
        const [px,py] = toCanvas(x,y);
        i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
      });
      ctx.stroke();
      const [lx,ly] = toCanvas(...path[path.length-1]);
      ctx.fillStyle = opt.color;
      ctx.beginPath(); ctx.arc(lx,ly,5,0,2*Math.PI); ctx.fill();
    });

    const [sx2,sy2] = toCanvas(...start);
    ctx.strokeStyle='#ffffff88'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(sx2,sy2,6,0,2*Math.PI); ctx.stroke();
    ctx.fillStyle='#ffffff66'; ctx.font='10px monospace'; ctx.textAlign='right';
    ctx.fillText('início', sx2-8, sy2+4);

    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    ctx.fillText('passo '+step+'/'+MAX_STEPS, W-8, 16);
  }

  window.optimRun = function() {
    running = !running;
    document.getElementById('optim-btn').textContent = running ? '⏸ Pausar' : '▶ Animar';
    if (running) animLoop();
    else cancelAnimationFrame(animId);
  };

  window.optimReset = function() {
    running = false; cancelAnimationFrame(animId); step = 0;
    document.getElementById('optim-btn').textContent = '▶ Animar';
    Object.values(optimizers).forEach(opt => {
      opt.pos = [...start];
      if (opt.v) opt.v = [0,0];
      if (opt.m) opt.m = [0,0];
      if (opt.v2) opt.v2 = [0,0];
      if ('t' in opt) opt.t = 0;
      opt.path = [[...start]];
    });
    drawFrame();
  };

  function animLoop() {
    if (step < MAX_STEPS && running) {
      stepOptimizers(); step++;
      drawFrame();
      animId = requestAnimationFrame(animLoop);
    } else {
      running = false;
      document.getElementById('optim-btn').textContent = '▶ Animar';
    }
  }

  drawFrame();
  window.addEventListener('resize', drawFrame);
})();
</script>

!!! info "Por que Adam converge mais rápido?"
    Adam adapta a taxa de aprendizado **por parâmetro**: dimensões com gradientes grandes recebem passos menores; dimensões com gradientes pequenos (como o eixo Y elíptico) recebem passos maiores. Isso corrige a anisotropia da superfície de perda automaticamente.

## AdamW — Adam com Weight Decay

**AdamW**[^5] separa o weight decay da atualização dos momentos, o que é matematicamente mais correto:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_t$$

onde $\lambda$ é o coeficiente de weight decay aplicado **diretamente** nos pesos (não no gradiente). É o padrão em treinamento de Transformers e LLMs.

## Recursos Adicionais

<iframe width="100%" height="470" src="https://www.youtube.com/embed/MD2fYip6QsQ" title="Who&#39;s Adam and What&#39;s He Optimizing?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

[^1]: [Introduction to Gradient Descent and Backpropagation Algorithm](https://atcold.github.io/NYU-DLSP20/en/week02/02-1/){:target="_blank"}, LeCun, Y.
[^2]: [ADAM: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980){:target="_blank"}, Kingma, D. P., & Ba, J.
[^3]: [Dive into Deep Learning](https://d2l.ai){:target="_blank"}, Zhang, A., & Lipton, Z. C.

---

--8<-- "docs/2026.2/classes/optimization/quiz.pt.md"
