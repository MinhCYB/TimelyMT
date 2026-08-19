# TimelyMT Vietnamese Overleaf Report

The report targets Overleaf's default **pdfLaTeX** compiler. No compiler change should be required.

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

4. Press **Recompile**.
5. Keep Overleaf's default compiler. Do not select XeLaTeX or LuaLaTeX.

The Vietnamese source is UTF-8 and uses the pdfLaTeX-compatible `inputenc`, T5 `fontenc`, and `babel` setup in `main.tex`. The document does not use `fontspec`, `polyglossia`, `minted`, shell escape, custom fonts, a custom `latexmkrc`, or SVG rendering. The five PNG files are the only figures referenced by `main.tex`.

SVG versions may optionally be uploaded for archival purposes, but they are not used during compilation. If a PNG is accidentally absent, the report displays a framed filename placeholder while preserving the figure caption and number.
