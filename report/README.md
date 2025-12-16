# Report (ACL-style LaTeX Template)

This folder contains an **ACL-style paper template** copied from:
`/Users/alireza/Code/Research/SuperS/693ca89a6d54ceffcbcf85e2/latex`

## Quickstart

From this directory:

```bash
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

Output: `acl_latex.pdf`

## Editing

- **Main file**: `acl_latex.tex`
- **Section files**: edit content in `Sections/*.tex` (these are `\input{...}` by `acl_latex.tex`)
- **Appendix**: `Appendix/dataset_details.tex`
- **Bibliography**: `bib.bib`
- **Mode switch**: in `acl_latex.tex`, change `\usepackage[review]{acl}` to `final` when you want the final camera-ready look.

## Folder layout

- `acl_latex.tex`: main paper entrypoint (pdfLaTeX)
- `acl_lualatex.tex`: alternative entrypoint (LuaLaTeX / XeLaTeX)
- `acl.sty`, `acl_natbib.bst`: ACL style files
- `Sections/`: section stubs
- `Appendix/`: appendix stubs
- `bib.bib`: bibtex database
- `acl_formatting.md`, `acl_upstream_README.md`: upstream guidelines (kept for reference)


