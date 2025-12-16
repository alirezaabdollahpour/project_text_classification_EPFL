# Report (LaTeX Template)

This folder contains a **4-page report template** for EPFL ML Project 2 (Text Classification).

## Quickstart

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## What to fill in

- Title must be **descriptive** (not “Project 2”), per the project description (`file:///Users/alireza/Code/EPFL/ML/Project02/project2_description.pdf`).
- Include **solid baselines** and **fair evaluation** (validation / CV), and an **Ethical risks** section (200–400 words) as required in the PDF.
- The template is structured to match the repo’s implementations:
  - `baseline_classifier.py`: hashed features / embedding-bag classifier
  - `glove_solution.py` (+ scripts): GloVe training pipeline
  - `distilbert_classifier.py`: DistilBERT fine-tuning, optional LoRA/SWA, optional fusion with averaged embeddings

## Folder layout

- `main.tex`: the paper template
- `refs.bib`: bibliography (add your own citations here)
- `figures/`: put plots here (kept in git via `.gitkeep`)


