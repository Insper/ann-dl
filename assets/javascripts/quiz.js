/**
 * ANN-DL Interactive Quiz Engine
 * Usage: buildQuiz('unique-id', 'Topic Title', questionsArray)
 * Each question: { q, opts: [string...], ans: index, exp: string }
 */
function buildQuiz(id, title, qs) {
  const wrap = document.getElementById('quiz-' + id);
  if (!wrap) return;

  const colors = {
    bg: '#0d1117', card: '#161b22', border: '#30363d',
    correct: '#1f3d1f', wrong: '#3d1a1a',
    correctText: '#3fb950', wrongText: '#ff7b72',
    neutral: '#c9d1d9', muted: '#8b949e', accent: '#f0883e'
  };

  wrap.style.cssText = `background:${colors.bg};border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;`;

  // Header
  const header = document.createElement('div');
  header.style.cssText = `display:flex;align-items:center;gap:.8rem;margin-bottom:1.2rem;padding-bottom:.8rem;border-bottom:1px solid ${colors.border};`;
  header.innerHTML = `
    <span style="font-size:1.3rem;">🧠</span>
    <div>
      <div style="color:${colors.accent};font-weight:bold;font-size:1rem;">Quiz — ${title}</div>
      <div style="color:${colors.muted};font-size:.8rem;">${qs.length} questões · clique em uma opção e depois em Verificar</div>
    </div>
    <div id="quiz-${id}-score-badge" style="margin-left:auto;display:none;padding:4px 14px;border-radius:20px;font-weight:bold;font-size:.9rem;"></div>
  `;
  wrap.appendChild(header);

  // Questions
  const qContainer = document.createElement('div');
  qs.forEach((q, qi) => {
    const qDiv = document.createElement('div');
    qDiv.id = `quiz-${id}-q${qi}`;
    qDiv.style.cssText = `margin-bottom:1rem;padding:1rem;background:${colors.card};border-radius:8px;border:1px solid ${colors.border};transition:border-color .2s;`;

    const qText = document.createElement('div');
    qText.style.cssText = `color:${colors.neutral};font-weight:600;margin-bottom:.7rem;font-size:.95rem;`;
    qText.textContent = `${qi + 1}. ${q.q}`;
    qDiv.appendChild(qText);

    const optsDiv = document.createElement('div');
    q.opts.forEach((opt, oi) => {
      const label = document.createElement('label');
      label.id = `quiz-${id}-q${qi}-l${oi}`;
      label.style.cssText = `display:flex;align-items:flex-start;gap:.6rem;padding:.45rem .6rem;border-radius:5px;cursor:pointer;margin-bottom:.2rem;transition:background .15s;border:1px solid transparent;`;
      label.onmouseover = () => { if (!label.dataset.locked) label.style.background = '#21262d'; };
      label.onmouseout = () => { if (!label.dataset.locked) label.style.background = 'transparent'; };
      label.innerHTML = `
        <input type="radio" name="quiz-${id}-q${qi}" value="${oi}" style="margin-top:3px;accent-color:${colors.accent};flex-shrink:0;">
        <span id="quiz-${id}-q${qi}-t${oi}" style="color:${colors.muted};font-size:.9rem;">${opt}</span>
      `;
      optsDiv.appendChild(label);
    });
    qDiv.appendChild(optsDiv);

    // Explanation placeholder
    const exp = document.createElement('div');
    exp.id = `quiz-${id}-q${qi}-exp`;
    exp.style.cssText = `display:none;margin-top:.6rem;padding:.6rem .8rem;background:#0d2137;border-left:3px solid #58a6ff;border-radius:0 5px 5px 0;color:#8b949e;font-size:.83rem;line-height:1.5;`;
    qDiv.appendChild(exp);

    qContainer.appendChild(qDiv);
  });
  wrap.appendChild(qContainer);

  // Submit button + result
  const footer = document.createElement('div');
  footer.style.cssText = 'display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.5rem;';
  footer.innerHTML = `
    <button id="quiz-${id}-btn" onclick="submitQuiz('${id}', ${JSON.stringify(qs.map(q=>q.ans))}, ${JSON.stringify(qs.map(q=>q.exp))})"
      style="padding:8px 26px;background:${colors.accent};color:#0d1117;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:.9rem;">
      Verificar Respostas
    </button>
    <button onclick="resetQuiz('${id}', ${JSON.stringify(qs.length)})"
      style="padding:8px 18px;background:#21262d;color:${colors.neutral};border:1px solid ${colors.border};border-radius:6px;cursor:pointer;font-size:.9rem;">
      ↺ Refazer
    </button>
    <div id="quiz-${id}-result" style="color:${colors.muted};font-size:.85rem;"></div>
  `;
  wrap.appendChild(footer);
}

window.submitQuiz = function(id, answers, explanations) {
  let score = 0;
  const btn = document.getElementById(`quiz-${id}-btn`);
  if (btn) btn.disabled = true;

  answers.forEach((ans, qi) => {
    const sel = document.querySelector(`input[name="quiz-${id}-q${qi}"]:checked`);
    const qDiv = document.getElementById(`quiz-${id}-q${qi}`);
    const exp = document.getElementById(`quiz-${id}-q${qi}-exp`);
    const chosen = sel ? parseInt(sel.value) : -1;
    const correct = chosen === ans;
    if (correct) score++;

    // Style options
    const numOpts = document.querySelectorAll(`input[name="quiz-${id}-q${qi}"]`).length;
    for (let oi = 0; oi < numOpts; oi++) {
      const label = document.getElementById(`quiz-${id}-q${qi}-l${oi}`);
      const txt = document.getElementById(`quiz-${id}-q${qi}-t${oi}`);
      const inp = label ? label.querySelector('input') : null;
      if (!label) continue;
      label.dataset.locked = '1';
      label.style.cursor = 'default';
      if (inp) inp.disabled = true;
      if (oi === ans) {
        label.style.background = '#1f3d1f';
        label.style.borderColor = '#3fb950';
        if (txt) txt.style.color = '#3fb950';
      } else if (oi === chosen && !correct) {
        label.style.background = '#3d1a1a';
        label.style.borderColor = '#ff7b72';
        if (txt) txt.style.color = '#ff7b72';
      }
    }

    // Border on question card
    if (qDiv) qDiv.style.borderColor = correct ? '#3fb950' : (chosen === -1 ? '#d29922' : '#ff7b72');

    // Show explanation
    if (exp && explanations[qi]) {
      exp.style.display = 'block';
      exp.innerHTML = `<strong style="color:#58a6ff;">💡 Explicação:</strong> ${explanations[qi]}`;
    }
  });

  // Score display
  const total = answers.length;
  const pct = Math.round(score / total * 100);
  const badge = document.getElementById(`quiz-${id}-score-badge`);
  const result = document.getElementById(`quiz-${id}-result`);
  const color = pct >= 80 ? '#3fb950' : pct >= 50 ? '#d29922' : '#ff7b72';
  const msg = pct >= 80 ? '🎉 Excelente!' : pct >= 50 ? '👍 Bom trabalho!' : '📚 Revise o conteúdo';

  if (badge) {
    badge.style.display = 'block';
    badge.style.background = color + '22';
    badge.style.color = color;
    badge.style.border = `1px solid ${color}`;
    badge.textContent = `${score}/${total}`;
  }
  if (result) result.innerHTML = `<span style="color:${color};font-weight:bold;">${pct}% — ${msg}</span>`;
};

window.resetQuiz = function(id, n) {
  for (let qi = 0; qi < n; qi++) {
    const numOpts = document.querySelectorAll(`input[name="quiz-${id}-q${qi}"]`).length;
    for (let oi = 0; oi < numOpts; oi++) {
      const label = document.getElementById(`quiz-${id}-q${qi}-l${oi}`);
      const txt = document.getElementById(`quiz-${id}-q${qi}-t${oi}`);
      const inp = label ? label.querySelector('input') : null;
      if (label) { label.style.background='transparent'; label.style.borderColor='transparent'; label.dataset.locked=''; label.style.cursor='pointer'; }
      if (txt) txt.style.color = '#8b949e';
      if (inp) { inp.checked = false; inp.disabled = false; }
    }
    const qDiv = document.getElementById(`quiz-${id}-q${qi}`);
    if (qDiv) qDiv.style.borderColor = '#30363d';
    const exp = document.getElementById(`quiz-${id}-q${qi}-exp`);
    if (exp) exp.style.display = 'none';
  }
  const btn = document.getElementById(`quiz-${id}-btn`);
  if (btn) btn.disabled = false;
  const badge = document.getElementById(`quiz-${id}-score-badge`);
  if (badge) badge.style.display = 'none';
  const result = document.getElementById(`quiz-${id}-result`);
  if (result) result.innerHTML = '';
};
