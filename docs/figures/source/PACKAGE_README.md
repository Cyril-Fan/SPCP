# SPCP structure diagrams

`figures/SPCP_overall.pdf` and `figures/SPCP_core.pdf` are 160 mm-wide vector
exports with embedded fonts. `editable/` contains their editable SVG sources
and the single-glyph font used for the image-alignment symbol.

The overall diagram presents fixed LLM-assisted class descriptions, complementary
prompt interaction, and the semantic and image-alignment constraints. In the
semantic diagram, prompted and reference difference vectors use the same class
pair; neighbor sets are constructed from fixed source-class reference features
before training. Denominators and averaging follow the manuscript definitions.

SVG text uses Times New Roman; the core neighbor-set symbol uses Cambria Math.
The image-alignment glyph matches newtxmath's calligraphic I. PDF use does not
require installing the supplied glyph font.
