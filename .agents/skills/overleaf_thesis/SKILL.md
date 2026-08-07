---
name: overleaf_thesis
description: Protocol for developing, formatting, validating, and editing a LaTeX thesis for Overleaf. Manages chapter files (chapters/), bibliography (references.bib), media inclusion, math/CFD/PINN notation, syntax checks, quality control reports, and scientific English editing.
---

# overleaf_thesis: Overleaf LaTeX Thesis Development & Quality Protocol

## 1. Overview & Scope

The `overleaf_thesis` skill guides Antigravity in editing, writing, validating, and organizing LaTeX thesis chapters for seamless integration with Overleaf.

### Scope Boundaries:
- **Managed Files**: 
  - `Latex/Chapters/1.Introduction.tex`
  - `Latex/Chapters/2.Fluid-dynamics.tex`
  - `Latex/Chapters/3.PINNs.tex`
  - `Latex/Chapters/4.Results.tex`
  - `Latex/Chapters/5.Conclusions.tex`
  - `Latex/references.bib`
  - `Latex/media/*` (figures, diagrams, plots)
- **Unmanaged File**: `main.tex` remains exclusively on Overleaf (or managed manually by the user). The skill provides preamble suggestions and `\input{}` / `\include{}` directives when requested.

---

## 2. Chapter Structure & File Organization

Each chapter must reside in `Latex/Chapters/` as a standalone `.tex` file using standard naming:

1. `Latex/Chapters/1.Introduction.tex` — Context, state of the art, thesis goals, contributions, thesis structure.
2. `Latex/Chapters/2.Fluid-dynamics.tex` — Fluid mechanics fundamentals, governing equations (Navier-Stokes, constitutive laws like Oldroyd-B), dimensionless numbers ($Re$, $Wi$, $\beta$), CFD principles.
3. `Latex/Chapters/3.PINNs.tex` — Literature review in 3 distinct parts: (1) General SciML & PINN overview vs traditional CFD/ML, (2) Foundational PINN framework by Raissi et al. (2019) (Automatic Differentiation, composite loss, forward vs. inverse problems), and (3) ViscoelasticNet paper by Thakur et al. (2024) (stress discovery from velocity inputs, HWNP robustness, model selection).
4. `Latex/Chapters/4.Results.tex` — Four-roll mill / viscoelastic flow results, quantitative field validation against COMSOL, loss progression, hyperparameter & staged training performance.
5. `Latex/Chapters/5.Conclusions.tex` — Summary of achievements, limitations of PINN formulations, future research perspectives.

### Chapter Integration Snippet (For User's `main.tex`):
```latex
% In Overleaf main.tex preamble:
\usepackage[utf8]{utf8}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{caption,subcaption}
\usepackage{siunitx}
\usepackage[style=ieee,backend=biber]{biblatex}
\addbibresource{references.bib}

% In main.tex document body:
\input{Chapters/2.Fluid-dynamics}
\input{Chapters/3.PINNs}
```

---

## 3. LaTeX Syntax & Quality Control Protocol

Whenever reviewing, editing, or validating thesis files, execute a **Quality Control (QC)** check.

### Syntax Check Rules:
1. **Balanced Environments**: Every `\begin{env}` must match an `\end{env}`.
2. **Brace Matching**: Ensure all curly braces `{}` and square brackets `[]` are strictly balanced.
3. **Hierarchy Integrity**: Verify logical structural depth (`\chapter{}` -> `\section{}` -> `\subsection{}` -> `\subsubsection{}`).
4. **Cross-Referencing**:
   - Labels: `\label{chap:...}`, `\label{sec:...}`, `\label{fig:...}`, `\label{tab:...}`, `\label{eq:...}`.
   - References: Use `\ref{sec:...}` for sections/figures, `\eqref{eq:...}` for equations, and `\cite{...}` for citations.

### QC Report Output Format:
When asked to perform a Quality Control audit, generate a report structured as follows:

```markdown
# LaTeX Quality Control Report

## Summary Status: [ OK | WARNING | ERROR ]

### 1. File Structure & Existence
- [x] chapters/01_introduction.tex (Valid)
- [x] chapters/02_fluidodynamic_background.tex (Valid)
- ...

### 2. Syntax & Environment Balance
- No unclosed environments found.
- Unmatched braces: None.

### 3. Bibliography & Citation Audit
- Total citations: N
- Unresolved \cite{} keys: [None or List]
- Duplicate BibTeX entries: [None or List]

### 4. Media & Figure Check
- Figures referenced in text before placement: Verified
- Missing image files: [None or List]

### 5. Actionable Recommendations
- Recommendation 1...
```

---

## 4. Bibliography Management (`references.bib`)

1. **Syntax Integrity**: Ensure BibTeX entries follow valid key-value structures (`@article`, `@book`, `@inproceedings`, `@phdthesis`).
2. **Citation Style**: IEEE or ACS style (`\usepackage[style=ieee]{biblatex}`).
3. **Consistency**:
   - Check that every `\cite{key}` used in `chapters/*.tex` exists in `references.bib`.
   - Remove duplicate keys or conflicting author/year definitions.
4. **Online Literature Integration**:
   - When missing references are detected or new papers need to be cited, query online academic databases (arXiv, DOI, PubMed, IEEE Xplore, Google Scholar).
   - **Mandatory User Notification**: Always notify the user before adding or updating entries in `references.bib`. Format entries using standard DOI-backed BibTeX.

---

## 5. Figures and Media Standards (`media/`)

1. **File Location**: Place all figures in `media/` (e.g., `media/4roll_streamlines.png`, `media/pinn_architecture.pdf`).
2. **Environment Template**:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\linewidth]{media/4roll_streamlines.png}
    \caption{Velocity vector fields and streamlines for the four-roll mill viscoelastic flow at $Wi = 0.5$.}
    \label{fig:4roll_streamlines}
\end{figure}
```
3. **Rules**:
   - Always include descriptive captions.
   - Always assign a unique `\label{fig:...}`.
   - Verify figure file existence on disk before referencing.
   - **Text Citation Rule**: Every figure MUST be explicitly introduced and discussed in the main body text (e.g., "As illustrated in Figure~\ref{fig:4roll_streamlines}, ...") *before* or alongside its visual appearance.

---

## 6. Mathematical & Physics Notation Guidelines (CFD & PINNs)

1. **Packages**: Use `amsmath`, `amssymb`, `siunitx`.
2. **SI Units**: Use `\SI{value}{unit}` or `\qty{value}{unit}` (e.g., `\SI{1.0}{\pascal\second}`, `\SI{1000}{\kilogram\per\cubic\meter}`).
3. **CFD & Viscoelastic Notation**:
   - Stream function satisfy incompressibility: $u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$.
   - Vectors/Tensors: Use bold symbols `\boldsymbol{u}`, `\boldsymbol{\tau}`, `\boldsymbol{\nabla}`.
   - Dimensionless numbers: Weissenberg number $Wi = \lambda \dot{\gamma}$, Reynolds number $Re = \frac{\rho U L}{\mu}$, viscosity ratio $\beta = \frac{\mu_s}{\mu_0}$.
4. **Numbered Equations**:
```latex
\begin{equation}
    \boldsymbol{\tau} + \lambda \left( \frac{\partial \boldsymbol{\tau}}{\partial t} + \boldsymbol{u} \cdot \boldsymbol{\nabla} \boldsymbol{\tau} - (\boldsymbol{\nabla}\boldsymbol{u})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \boldsymbol{\nabla}\boldsymbol{u} \right) = 2 \mu_p \boldsymbol{D}
    \label{eq:oldroyd_b}
\end{equation}
```
5. **Multi-line Equations**: Use `align` or `subequations` for coupled PINN loss components ($L_{res}$, $L_{bc}$, $L_{data}$).

---

## 7. Scientific English Writing & Style Guidelines

When writing or editing thesis text, adhere strictly to these scientific writing principles:

1. **American English**: Consistent spelling (e.g., *viscoelasticity*, *behavior*, *modeling*, *streamline*).
2. **Concise Sentences**: Keep average sentence length under 25 words. Split compound sentences.
3. **Active Voice**: Prefer active voice where appropriate (e.g., "We formulate the residual loss..." rather than "The residual loss is formulated by us..."), maintaining objective scientific tone.
4. **Paragraph Structure**:
   - One main idea per paragraph.
   - Topic sentence followed by supporting data/explanations and logical transition sentences.
5. **Acronym & Symbol Definitions**:
   - Spell out acronyms on first occurrence in the text: e.g., *Physics-Informed Neural Networks (PINNs)*, *Phan-Thien-Tanner (PTT)*, *Computational Fluid Dynamics (CFD)*.
   - Maintain a consistent symbol notation across all chapters.
6. **No Informalities**: Avoid contractions (*don't*, *can't*), colloquialisms, or vague qualifiers (*a lot of*, *very good*). Use precise quantitative descriptors (*a 15\% reduction in L2 relative error*).

---

## 8. Step-by-Step Skill Workflow

When invoked by the user (e.g., "Antigravity, review chapter 2", "Write introduction for LaTeX thesis", "Run quality check on thesis files"):

1. **Inspect Context & Files**:
   - Check existing files in `chapters/`, `references.bib`, and `media/`.
   - Read relevant codebase details (e.g., `final_roll/src/physics.py` or `PINN-wiki`) if writing technical sections.

2. **Execute Action**:
   - **Drafting/Editing**: Apply scientific English rules, math notation standards, and proper sectioning.
   - **Bibliography**: Check or insert BibTeX citations, verifying against `references.bib`.
   - **Figures**: Format LaTeX figure blocks and check file existence in `media/`.

3. **Run Quality Control**:
   - Perform LaTeX syntax check (balanced environment tags, brace counts, citation links).
   - Generate summary report if requested.

4. **User Communication**:
   - Report changes cleanly with file links.
   - Highlight any open questions or required user actions for Overleaf `main.tex`.
