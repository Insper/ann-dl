# 2026.2 — Artificial Neural Networks and Deep Learning

## Instructor

| :fontawesome-regular-address-book: | :fontawesome-regular-envelope: |
|-|-:|
| [Humberto Sandmann](https://hsandmann.github.io){target='_blank'} | [humbertors@insper.edu.br](mailto:humbertors@insper.edu.br){target='_blank'} |

## Schedule

| :octicons-location-24: | :fontawesome-regular-calendar: | :fontawesome-regular-clock: |
|-|:-:|:-:|
| Lecture | - | -h00 :fontawesome-solid-arrow-right-long: -h00 |
| Lecture | - | -h00 :fontawesome-solid-arrow-right-long: -h00 |
| Office Hours | - | -h00 :fontawesome-solid-arrow-right-long: -h30 |

## Final Grade

$$
\text{Final} = \left\{\begin{array}{lll}
    \text{Individual} \geq 5 \bigwedge \text{Team} \geq 5 &
    \implies &
    \displaystyle \frac{ \text{Individual} + \text{Team} } {2}
    \\
    \\
    \text{Otherwise} &
    \implies &
    \min\left(\text{Individual}, \text{Team}\right)
    \end{array}\right.
$$

---

## Syllabus 2026.2

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1.2rem;margin:2rem 0;">

<div style="background:linear-gradient(160deg,#1a365d,#2c5282);border-radius:14px;padding:1.2rem 1.2rem 1.2rem 0.8rem;color:#fff;">
  <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:2px;opacity:.6;margin-bottom:.25rem;">Module 1</div>
  <div style="font-size:1.05rem;font-weight:700;border-bottom:1px solid rgba(255,255,255,.18);padding-bottom:.6rem;margin-bottom:.8rem;">Foundations</div>
  <ul style="list-style:none;padding:0;margin:0;font-size:.8rem;line-height:1.85;">
    <li>Concepts &amp; AI</li>
    <li>Data</li>
    <li>Preprocessing</li>
    <li>Neural Networks</li>
    <li>Perceptron</li>
    <li>MLP</li>
    <li>Optimization</li>
    <li>Regularization</li>
    <li>Metrics</li>
  </ul>
</div>

<div style="background:linear-gradient(160deg,#44337a,#6b46c1);border-radius:14px;padding:1.2rem 1.2rem 1.2rem 0.8rem;color:#fff;">
  <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:2px;opacity:.6;margin-bottom:.25rem;">Module 2</div>
  <div style="font-size:1.05rem;font-weight:700;border-bottom:1px solid rgba(255,255,255,.18);padding-bottom:.6rem;margin-bottom:.8rem;">Deep Architectures</div>
  <ul style="list-style:none;padding:0;margin:0;font-size:.8rem;line-height:1.85;">
    <li>DL Layers</li>
    <li>CNNs</li>
    <li>Attention <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
    <li>Transformers <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
    <li>Transfer Learning <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
    <li>LLMs <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
  </ul>
</div>

<div style="background:linear-gradient(160deg,#1a4731,#276749);border-radius:14px;padding:1.2rem 1.2rem 1.2rem 0.8rem;color:#fff;">
  <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:2px;opacity:.6;margin-bottom:.25rem;">Module 3</div>
  <div style="font-size:1.05rem;font-weight:700;border-bottom:1px solid rgba(255,255,255,.18);padding-bottom:.6rem;margin-bottom:.8rem;">Generative Models</div>
  <ul style="list-style:none;padding:0;margin:0;font-size:.8rem;line-height:1.85;">
    <li>Overview</li>
    <li>VAE</li>
    <li>GAN</li>
    <li>CLIP</li>
    <li>Stable Diffusion</li>
    <li>Flow Matching</li>
    <li>Diffusion Transformers <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
    <li>AR Generation <span style="background:rgba(255,255,255,.2);border-radius:6px;padding:1px 7px;font-size:.68rem;font-weight:700;letter-spacing:.5px;vertical-align:middle;">NEW</span></li>
  </ul>
</div>

</div>

---

## Course Description

This course provides a comprehensive introduction to Artificial Neural Networks and Deep Learning, using modern frameworks (primarily PyTorch). Topics span mathematical foundations, core architectures (MLPs, CNNs, Transformers), attention mechanisms, generative models (GANs, VAEs, Diffusion, Flow Matching), Diffusion Transformers, and Large Language Models. Equal emphasis is placed on theoretical understanding and practical application.

## Learning Objectives

By the end of this course, students will be able to:

1. **Understand Fundamentals**: explain gradient descent, backpropagation, activation functions, and regularization.
2. **Master Key Architectures**: describe and motivate MLPs, CNNs, Transformers, and LLMs.
3. **Implement with PyTorch**: train and debug deep learning models.
4. **Evaluate Performance**: apply appropriate metrics and regularization/optimization techniques.
5. **Work with Generative Models**: understand and apply GANs, VAEs, Diffusion, and Flow Matching.
6. **Apply Transfer Learning**: fine-tune pre-trained models using PEFT techniques (LoRA, QLoRA).
7. **Understand LLMs**: grasp architecture, training (RLHF, DPO), and applications of Large Language Models.
8. **Critically Evaluate Research**: read and assess current deep learning papers.

## Bibliography

**Core:**

1. Fleuret, F. (2023). [The Little Book of Deep Learning](https://fleuret.org/lbdl){:target="_blank"}.
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/){:target="_blank"}. MIT Press.

**Supplementary:**

1. Nielsen, M. A. (2019). [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/){:target="_blank"}.
1. Zhang, A. et al. (2024). [Dive into Deep Learning](https://d2l.ai/){:target="_blank"}.
1. Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762){:target="_blank"}.
1. Brown, T. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165){:target="_blank"}.
