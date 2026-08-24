# LaTeX report

This directory contains the reproducible diabetes-classifier course report. Its cover and preliminary pages follow the supplied Tribhuvan University/Purwanchal Campus project-report layout, while the main paper keeps the supplied research paper's organization and numbered citation style. The prose, tables, and figures were prepared specifically for this project.

The three approved authors and institutional details are populated in `report.tex`. The report uses A4 paper and a Times-style academic typeface. The cover is unnumbered, the abstract, table of contents, and abbreviation list use Roman numerals, and the main report restarts at Arabic page 1. Separate lists of figures and tables are intentionally omitted.

## Build and audit

Requirements:

- the project dependencies installed through `uv`;
- Tectonic 0.17 or a compatible LaTeX installation; and
- Poppler's `pdftotext`, `pdfinfo`, and `pdffonts` for the local report audit.

From the repository root, run:

```bash
make -C src-tex
```

This command:

1. reruns the exact train/test experiment;
2. verifies the expected metrics;
3. regenerates all seven PDF figures and the results table;
4. compiles `src-tex/report.pdf`; and
5. checks citations, numerical consistency, render warnings, placeholders, and exact phrase overlap with the supplied sample and instructions.

The phrase-overlap check is a small local safeguard, not a substitute for institutional plagiarism or authorship review, and it cannot guarantee an institutional checker result. No report text is uploaded to a third-party service.

## Main files

```text
report.tex                 Main manuscript
references.bib             Cited sources
report.pdf                 Reviewed rendered report
assets/tu-logo.png         TU logo used on the cover
generate_figures.py        Reproducible experiment and plots
check_report.py            Local citation/consistency/similarity audit
figures/                    Generated vector figures
generated/results.json     Machine-readable rerun results
generated/model_metrics.*  Generated results table
sn-jnl.cls                 Official Springer Nature class
sn-vancouver-num.bst       Official numbered reference style
```

The Springer class and bibliography-style files are unchanged assets from Springer Nature's LaTeX package, version 3.1 (December 2024): <https://www.springernature.com/gp/authors/campaigns/latex-author-support>. The cover logo was extracted from the user-supplied university report reference. The committed PDF is formatted as a course report rather than a journal-submission package.
