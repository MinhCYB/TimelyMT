# TimelyMT Vietnamese Overleaf Report

The report targets Overleaf's default **pdfLaTeX** compiler. No compiler change should be required.

The document uses the 11pt A4 `article` class so the nine numbered research sections flow continuously instead of forcing thesis-style chapter openings. Figures and tables remain numbered within their top-level section, and the three appendices may share pages. Full page breaks are reserved for the title/front matter boundary, the main body, the bibliography, and the appendix block.

## Upload

1. Create a blank Overleaf project.
2. Upload `main.tex` and `references.bib` to the project root.
3. Upload the entire existing figure directory so the Overleaf project has this structure:

   ```text
   main.tex
   references.bib
   figures/
     figure-1-streaming-pipeline.png
     figure-2-policy-feature-evolution.png
     figure-3-controlled-ablation.png
     figure-4-divergence-cascade.png
     figure-5-interactive-demo.png
   ```

4. Open **Logs and output files**, choose **Clear cached files**, then select **Recompile from scratch**. This forces Overleaf to regenerate `main.aux`, run BibTeX, and rebuild `main.bbl` instead of reusing the stale empty bibliography cache.
5. Keep Overleaf's default compiler. Do not select XeLaTeX or LuaLaTeX.

The Vietnamese source is UTF-8 and uses the pdfLaTeX-compatible `inputenc`, T5 `fontenc`, and `babel` setup in `main.tex`. The document does not use `fontspec`, `polyglossia`, `minted`, shell escape, custom fonts, a custom `latexmkrc`, or SVG rendering. The five PNG files are the only figures referenced by `main.tex`.

The bibliography uses `natbib` with the author-year `plainnat` style and exactly the eight entries in `references.bib`. A manual local build, when TeX tools are installed, is:

```text
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

When building from the repository rather than an uploaded Overleaf project, run these commands from `reports/overleaf/` so the `../figures/` fallback resolves all five PNG files.

After the first pdfLaTeX pass, `main.aux` must contain `\citation{...}`, `\bibstyle{plainnat}`, and `\bibdata{references}`. After BibTeX, `main.bbl` must exist and contain eight `\bibitem` entries. If the references page is still empty, confirm that `references.bib` is beside `main.tex` at the Overleaf project root, clear cached files again, and inspect the BibTeX section of the log for a missing database or style file. `main.tex` now fails loudly when `references.bib` is absent and explicitly requests all eight approved entries with `\nocite`.

SVG versions may optionally be uploaded for archival purposes, but they are not used during compilation. If a PNG is accidentally absent, the report displays a framed filename placeholder while preserving the figure caption and number.
