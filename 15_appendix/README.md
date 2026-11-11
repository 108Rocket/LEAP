# Supplementary Proofs

[Read the PDF](LEAP_Appendix.pdf) | [LaTeX source](LEAP_Appendix.tex) | [README proofs](../README.md#appendix)

Appendices A--D cover the cross-field bound, critical-position proximity,
weighted DTW block bound, and DTW signal lower bound.

The PDF uses `IEEEtran` conference format: two columns, 10-point text, US Letter
paper, standard margins, and embedded fonts. It is a repository companion,
not an appendix to append to the ICDE submission. The
[ICDE 2027 research-track instructions](https://icde2027.github.io/cf-research-papers)
do not allow appendices in submitted papers.

Rebuild with a TeX distribution that includes `IEEEtran`:

```bash
cd 15_appendix
latexmk -pdf -interaction=nonstopmode -halt-on-error LEAP_Appendix.tex
```

Keep the PDF and source in sync with the README proofs.
