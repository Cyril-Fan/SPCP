# Structure-figure export

The editable sources are `../SPCP_overall.svg` and `../SPCP_core.svg`.
`../fonts/SPCPNewTXMathI.otf` contains the single image-alignment glyph matching
the manuscript's `newtxmath` calligraphic I. The SVG also embeds this glyph font.
No fonts are installed or modified by the exporter.

This optional figure-authoring environment is separate from the Python 3.8.20
experiment environment. It needs Python 3.10 or later, the packages below,
Cairo and FreeType libraries, and Times New Roman and Cambria Math fonts.
The released training dependencies and protocols are unchanged.

```sh
python -m pip install -r docs/figures/source/requirements.txt
python docs/figures/source/build.py --output figure_export
```

Use a new or empty output directory. The script generates two vector PDFs,
their editable sources and glyph font, a short README, and
`SPCP_structure_figures.zip`. It does not overwrite repository figures.
PNGs in the parent directory are screen previews rendered from the PDFs.

On Windows, `CAIROCFFI_DLL_DIRECTORIES` may point to an existing Cairo/FreeType
DLL directory. If needed, specify the libraries directly using
`SPCP_CAIRO_LIBRARY` and `SPCP_FREETYPE_LIBRARY`. The exporter does not download
libraries, change system font configuration or access experiment data.
