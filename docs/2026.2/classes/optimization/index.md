
Gradient Descent is an optimization algorithm used to minimize the loss function in machine learning models. It works by iteratively adjusting the model parameters in the direction of the steepest descent of the loss function, as defined by the negative gradient.

The main idea behind gradient descent is to update the model parameters in the opposite direction of the gradient of the loss function with respect to the parameters. This is done using the following update rule:

\[
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
\]
/// caption
Vanilla Gradient Descent
///

where:

- \(\theta\) represents the model parameters,
- \(\eta\) is the learning rate (a hyperparameter that controls the step size),
- \(\nabla J(\theta)\) is the gradient of the loss function with respect to the parameters.

In the standard Supervised Learning paradigm, the loss (per sample) is simply the output of the cost function. Machine Learning is mostly about optimizing functions (usually minimizing them). It could also involve finding Nash Equilibria between two functions like with GANs. This is done using Gradient Based Methods, though not necessarily Gradient Descent[^1].

A **Gradient Based Method** is a method/algorithm that finds the minima of a function, assuming that one can easily compute the gradient of that function. It assumes that the function is continuous and differentiable almost everywhere (it need not be differentiable everywhere)[^1].

**Gradient Descent Intuition** - Imagine being in a mountain in the middle of a foggy night. Since you want to go down to the village and have only limited vision, you look around your immediate vicinity to find the direction of steepest descent and take a step in that direction[^1].

To visualize the gradient descent process, we can create a simple 3D plot that shows how the parameters of a model are updated over iterations to minimize a loss function. Below is an example code using Python with Matplotlib to create such a visualization.


``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/optimization/gradient-descent-path.py"
```

## Gradient Descent Variants

There are several variants of gradient descent, each with its own characteristics:

1. **Batch Gradient Descent**: Computes the gradient using the entire dataset. It provides a stable estimate of the gradient but can be slow for large datasets. Formula is given by:

    \[
    \theta = \theta - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla J(\theta; x^{(i)}, y^{(i)})
    \]

    ``` python
    for epoch in range(num_epochs):
        gradients = compute_gradients(X, y, model)
        model.parameters -= learning_rate * gradients
    ```

2. **Stochastic Gradient Descent (SGD)**: Computes the gradient using a single data point at a time. It introduces noise into the optimization process, which can help escape local minima but may lead to oscillations.  Formula is given by:

    \[
    \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t; x^{(i)}, y^{(i)})
    \]

    ``` python
    for epoch in range(num_epochs):
        for i in range(len(X)):
            gradients = compute_gradients(X[i], y[i], model)
            model.parameters -= learning_rate * gradients
    ```

3. **Mini-Batch Gradient Descent**: Combines the advantages of batch and stochastic gradient descent by using a small random subset of the data (mini-batch) to compute the gradient. It balances the stability of batch gradient descent with the speed of SGD.

    \[
    \theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i=1}^{B} \nabla J(\theta_t; x^{(i)}, y^{(i)})
    \]

    ``` python
    for epoch in range(num_epochs):
        for batch in get_mini_batches(X, y, batch_size):
            gradients = compute_gradients(batch.X, batch.y, model)
            model.parameters -= learning_rate * gradients
    ```


## Momentum

Momentum is a variant of gradient descent that helps accelerate the optimization process by using the past gradients to smooth out the updates. It introduces a "momentum" term that accumulates the past gradients and adds it to the current gradient update. The update rule with momentum is given by:

In Momentum, we have two iterates ($p$ and $\theta$) instead of just one. The updates are as follows:

$$
V_{t+1} = \beta V_t + (1 - \beta) \nabla J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta V_{t+1}
$$

$V$ is called the SGD momentum. At each update step we add the stochastic gradient to the old value of the momentum, after dampening it by a factor $\beta$ (value between 0 and 1). $V$ can be thought of as a running average of the gradients. Finally we move $\theta$ in the direction of the new momentum $V$ [^1].

Alternate Form: Stochastic Heavy Ball Method

\[
\theta_{t+1} = \theta_t - \eta \nabla f_i(\theta_t) + \beta( \theta_t - \theta_{t-1} ) \quad 0 \leq \beta < 1
\]

This form is mathematically equivalent to the previous form. Here, the next step is a combination of previous step’s direction and the new negative gradient.

The momentum term helps to dampen oscillations and can lead to faster convergence, especially in scenarios with noisy gradients or ravines in the loss landscape.


## RMSProp

RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to address the diminishing learning rates of AdaGrad. It maintains a moving average of the squared gradients and uses this to normalize the gradients. The update rule is given by:

$$
V_{t+1} = \beta V_t + (1 - \beta) \nabla (\theta_t)^2
$$

\[
\theta_{t+1} = \theta_t - \eta\frac{\nabla (\theta_t)}{\sqrt{V_{t+1} + \epsilon}}
\]

Where:

- \(V_t\) is the moving average of the squared gradients,
- \(\epsilon\) is a small constant added for numerical stability, helping to prevent division by zero.


<!-- ## AdaGrad

AdaGrad (Adaptive Gradient Algorithm) is an optimization algorithm that adapts the learning rate for each parameter based on the historical gradients. It performs larger updates for infrequent parameters and smaller updates for frequent parameters. The update rule is given by:

$$
V_{t+1} = \beta V_t + \nabla (\theta_t)^2
$$

\[
\theta_{t + 1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla J(\theta)
\]

Where:

- \(G_t\) is the diagonal matrix of the accumulated squared gradients,
- \(\epsilon\) is a small constant added for numerical stability. -->

## ADAM Optimizer

ADAM (Adaptive Moment Estimation) is an advanced optimization algorithm that combines the benefits of both AdaGrad and RMSProp. It maintains two moving averages for each parameter: the first moment (mean) and the second moment (uncentered variance). The update rules are as follows:

1. Compute the gradients:

\[
g_t = \nabla J(\theta_t)
\]

2. Update the first moment estimate:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]

3. Update the second moment estimate:

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]

4. Compute bias-corrected estimates:

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\]

\[
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

5. Update the parameters:

\[
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

Where:

- \(m_t\) is the first moment (the mean of the gradients),
- \(v_t\) is the second moment (the uncentered variance of the gradients),
- \(\beta_1\) and \(\beta_2\) are hyperparameters that control the decay rates of the moving averages,
- \(\epsilon\) is a small constant added for numerical stability.
The default values for the hyperparameters are typically set to \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), and \(\epsilon = 10^{-8}\)[^2][^3].

ADAM is widely used in practice due to its adaptive learning rate properties and is particularly effective for training deep learning models.



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

  // Loss surface: elongated ellipse (ill-conditioned)
  // f(x,y) = 0.1*x^2 + 2*y^2   minimum at (0,0)
  // gradient: [0.2x, 4y]
  function loss(x,y) { return 0.15*x*x + 2.5*y*y; }
  function grad(x,y) { return [0.3*x, 5*y]; }

  const start = [3.5, 2.8];
  const eta = 0.08;

  // Optimizers state
  const optimizers = {
    sgd: { pos: [...start], color: '#ff7b72', path: [[...start]] },
    momentum: { pos: [...start], v: [0,0], color: '#58a6ff', path: [[...start]] },
    adam: { pos: [...start], m: [0,0], v2: [0,0], t: 0, color: '#3fb950', path: [[...start]] },
    rmsprop: { pos: [...start], v2: [0,0], color: '#d29922', path: [[...start]] },
  };

  function stepOptimizers() {
    // SGD
    const gs = grad(...optimizers.sgd.pos);
    optimizers.sgd.pos[0] -= eta * gs[0];
    optimizers.sgd.pos[1] -= eta * gs[1];
    optimizers.sgd.path.push([...optimizers.sgd.pos]);

    // Momentum
    const gm = grad(...optimizers.momentum.pos);
    optimizers.momentum.v[0] = 0.9*optimizers.momentum.v[0] - eta*gm[0];
    optimizers.momentum.v[1] = 0.9*optimizers.momentum.v[1] - eta*gm[1];
    optimizers.momentum.pos[0] += optimizers.momentum.v[0];
    optimizers.momentum.pos[1] += optimizers.momentum.v[1];
    optimizers.momentum.path.push([...optimizers.momentum.pos]);

    // Adam
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

    // RMSProp
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

    // Background contours
    ctx.fillStyle = '#161b22'; ctx.fillRect(0,0,W,H);

    const levels = [0.1, 0.5, 1.5, 3.5, 7, 12, 18];
    levels.forEach((l, li) => {
      const alpha = 0.15 + li * 0.05;
      ctx.strokeStyle = `rgba(88,166,255,${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      // Ellipse: 0.15x²+2.5y²=l → x=sqrt(l/0.15)cosθ, y=sqrt(l/2.5)sinθ
      const a = Math.sqrt(l/0.15), b = Math.sqrt(l/2.5);
      for (let i=0;i<=100;i++) {
        const angle = i/100*2*Math.PI;
        const [px,py] = toCanvas(a*Math.cos(angle), b*Math.sin(angle));
        i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
      }
      ctx.stroke();
    });

    // Minimum marker
    const [mx0,my0] = toCanvas(0,0);
    ctx.strokeStyle='#ffffff44'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(mx0,my0,8,0,2*Math.PI); ctx.stroke();
    ctx.fillStyle='#ffffff66'; ctx.font='10px monospace'; ctx.textAlign='left';
    ctx.fillText('min', mx0+10, my0+4);

    // Draw paths
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
      // Current position
      const [lx,ly] = toCanvas(...path[path.length-1]);
      ctx.fillStyle = opt.color;
      ctx.beginPath(); ctx.arc(lx,ly,5,0,2*Math.PI); ctx.fill();
    });

    // Start marker
    const [sx2,sy2] = toCanvas(...start);
    ctx.strokeStyle='#ffffff88'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(sx2,sy2,6,0,2*Math.PI); ctx.stroke();
    ctx.fillStyle='#ffffff66'; ctx.font='10px monospace'; ctx.textAlign='right';
    ctx.fillText('start', sx2-8, sy2+4);

    // Step counter
    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    ctx.fillText('step '+step+'/'+MAX_STEPS, W-8, 16);
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

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_t
$$

onde $\lambda$ é o coeficiente de weight decay aplicado **diretamente** nos pesos (não no gradiente).

No Adam original, weight decay é adicionado ao gradiente antes do scaling adaptativo — o que distorce o decay. AdamW corrige isso. É o padrão em treinamento de Transformers e LLMs (GPT-2, BERT, LLaMA etc.).

## Additional Resources

<iframe width="100%" height="470" src="https://www.youtube.com/embed/MD2fYip6QsQ" title="Who&#39;s Adam and What&#39;s He Optimizing? | Deep Dive into Optimizers for Machine Learning!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>




[^1]: [Introduction to Gradient Descent and Backpropagation Algorithm](https://atcold.github.io/NYU-DLSP20/en/week02/02-1/){:target="_blank"}, LeCun, Y.

[^2]: [ADAM: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980){:target="_blank"}, Kingma, D. P., & Ba, J.

[^3]: [Dive into Deep Learning](https://d2l.ai){:target="_blank"}, Zhang, A., & Lipton, Z. C.

[^4]: [Stochastic and Mini-batch Gradient Descent](https://kenndanielso.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_6_Stochastic_and_minibatch_gradient_descent.html){:target="_blank"}

---

--8<-- "docs/2026.2/classes/optimization/quiz.md"
