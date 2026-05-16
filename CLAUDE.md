# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Course material site for **Artificial Neural Networks and Deep Learning** at Insper, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). All content lives under `docs/` and is published to GitHub Pages via CI.

## Commands

Activate the virtual environment first (required every session):

```shell
source ./env/bin/activate
```

| Task | Command |
|------|---------|
| Preview site locally | `mkdocs serve -o` |
| Deploy to GitHub Pages | `mkdocs gh-deploy` |
| Install dependencies | `pip3 install -r requirements.txt` |

The `MKDOCS_GIT_COMMITTERS_APIKEY` env var (GitHub PAT) is needed for the git-committers plugin; it is loaded from `.env` automatically by mkdocs when present.

## Architecture

```
docs/
  2026.2/                   # Current semester (21 classes including new Attention, Transformers, Transfer Learning, LLMs)
    classes/<topic>/        # One folder per lecture topic
      index.md              # Main lecture page (Markdown + MkDocs extensions + inline HTML/JS)
      *.py                  # Python scripts rendered inline via markdown-exec
      *.gif / *.png         # Figures referenced in pages
    exercises/<topic>/      # Student exercise pages per topic
    projects/               # Project briefs
    index.md                # Semester overview
  2025.2/                   # Previous semester (17 classes, archived)
    classes/<topic>/
    exercises/<topic>/
    projects/
    index.md
  definitions/              # Short comparison/definition pages
  biblio/                   # PDF copies of seminal papers
  assets/                   # CSS, JS, images
mkdocs.yml                  # Full site config: nav, plugins, markdown extensions
requirements.txt            # Python deps for both mkdocs plugins and code generation
```

## 2026.2 new classes

Compared to 2025.2, the following classes are new in 2026.2:
- `docs/2026.2/classes/attention/` — Mecanismos de Atenção (interactive JS heatmap, playground)
- `docs/2026.2/classes/transformers/` — Transformers (step-through architecture viz)
- `docs/2026.2/classes/transfer-learning/` — Transfer Learning & Fine-Tuning (LoRA, QLoRA)
- `docs/2026.2/classes/llms/` — Grandes Modelos de Linguagem (LLMs, RLHF, MoE)
- `docs/2026.2/classes/flow-matching/` — Rewritten with full content and animated particle viz

## Key MkDocs extensions in use

- **`markdown-exec`** — Python code blocks with `exec` fence execute at build time; output is embedded in the page. Python scripts in `docs/classes/*/` are often included this way.
- **`pymdownx.arithmatex` + MathJax** — LaTeX math via `$...$` and `$$...$$`.
- **`pymdownx.superfences`** — Mermaid diagrams (```` ```mermaid ````) and Plotly charts (```` ```plotly ````).
- **`encryptcontent`** — Password-protects selected pages; cache stored in `encryptcontent.cache`.
- **`termynal`** — Animated terminal blocks using `$` or `>` prompt prefixes.
- **`neoteroi.timeline`** — Timeline blocks for schedule pages.

## Content conventions

- Lecture pages (`docs/classes/<topic>/index.md`) mix prose, MathJax equations, Mermaid diagrams, and inline Python-generated figures.
- Python scripts that generate figures/animations live alongside `index.md` in the same topic folder and are executed at build time via `markdown-exec`.
- The `nav:` section in `mkdocs.yml` is the authoritative page order; adding a new page requires an entry there.
- The site is in **Portuguese** (Brazilian); new content should match the existing language of each page.
