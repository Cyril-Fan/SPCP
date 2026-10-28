# Structure-figure export

The editable sources are `../SPCP_overall.svg` and `../SPCP_core.svg`.
NewTX mathematical labels are genuine LaTeX-generated vector paths, with their
expressions retained in `math_labels.json` and in each SVG math group's
`data-latex` attribute. Ordinary labels and diagram geometry remain editable SVG.
`../fonts/SPCPNewTXMathI.otf` supplies the image-alignment glyph; the SVG embeds it.
No fonts are installed or modified by these tools.

## Export existing artwork

This optional authoring environment is separate from the Python 3.8.20 experiment
environment. It needs Python 3.10 or later, Cairo and FreeType libraries, and
Times New Roman and Cambria Math fonts. Training dependencies are unchanged.

```sh
python -m pip install -r docs/figures/source/requirements.txt
python docs/figures/source/build.py --output figure_export
```

Use a new or empty output directory. The exporter creates two vector PDFs,
editable SVGs, the glyph font, a short README and `SPCP_structure_figures.zip`.
It does not overwrite repository figures. PNGs are previews rendered from PDFs.
The existing SVGs already contain their math paths; TeX is not needed for export.

## Edit mathematical labels

Edit expressions and placement in `math_labels.json`, then regenerate the SVG
math groups before exporting. This needs Tectonic and its standard TeX packages
(`amsmath`, `amssymb`, `newtxtext`, `newtxmath`, `bm`, `xcolor`, `geometry`).

```sh
python docs/figures/source/regenerate_math.py --output math_export --tectonic tectonic
python docs/figures/source/build.py --output figure_export
```

`regenerate_math.py` updates only math groups in the two SVG sources; it does not
read experimental data. `math_renderer.py` renders the expressions and records
their baselines. Use `--only-cached` only when the TeX cache is already populated.
An optional `--class-file` path selects an available Springer class instead of
standard `article` when matching that typesetting environment.

On Windows, `CAIROCFFI_DLL_DIRECTORIES` may point to an existing Cairo/FreeType
DLL directory. If needed, specify `SPCP_CAIRO_LIBRARY` and
`SPCP_FREETYPE_LIBRARY` explicitly. The SVG exporter does not download libraries,
change system font configuration or access experiment data.
