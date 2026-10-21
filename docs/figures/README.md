# Paper figures / 论文图件

Figures synchronized with the current SPCP working manuscript (21 October 2026). PNG files provide previews, SVG files retain editable figure sources, and PDF files provide the versions used by the manuscript where applicable. The core-constraint PDF contains the high-resolution image; the overall-framework and analysis PDFs retain vector content.

本目录与当前 SPCP 稿件图件保持一致。PNG 用于预览，SVG 为可编辑原稿；总体框架与三张分析图另提供矢量 PDF。同步图件不改变实验数据、方法或训练配置。

| Manuscript figure | Content | Files |
|---|---|---|
| Fig. 1 | SPCP framework / 总体框架 | [PNG](SPCP_overall.png) · [SVG](SPCP_overall.svg) · [PDF](SPCP_overall.pdf) |
| Fig. 2 | Text semantic constraint / 文本语义约束 | [PNG](SPCP_core.png) · [SVG](SPCP_core.svg) · [PDF](SPCP_core.pdf) |
| Fig. 3 | Directional distortion / 方向畸变 | [PNG](Fig3_direction_distortion.png) · [SVG](Fig3_direction_distortion.svg) · [PDF](Fig3_direction_distortion.pdf) |
| Fig. 4 | Length distortion / 长度畸变 | [PNG](Fig4_length_distortion.png) · [SVG](Fig4_length_distortion.svg) · [PDF](Fig4_length_distortion.pdf) |
| Fig. 5 | Class-wise recognition changes / 类别识别变化 | [PNG](Fig5_classwise_errors.png) · [SVG](Fig5_classwise_errors.svg) · [PDF](Fig5_classwise_errors.pdf) |

## Analysis figures

![Directional distortion on base and new classes](Fig3_direction_distortion.png)

![Length distortion on base and new classes](Fig4_length_distortion.png)

![Class-wise recognition changes on NWPU-RESISC45 and PatternNet](Fig5_classwise_errors.png)

Full and w/o S differ jointly in the absolute and interclass text losses. New-class graphs are used solely for analysis, not training. Closer reference geometry does not imply uniform recognition gains: Full increases NWPU-RESISC45 new-class accuracy by 5.30 percentage points and decreases PatternNet new-class accuracy by 3.67 points relative to w/o S.

Full 与 w/o S 同时涉及绝对锚定和类间约束两项损失。新类图只用于分析，不用于训练。参考几何更接近不代表识别率普遍提高：相对 w/o S，Full 的 NWPU-RESISC45 新类准确率提高 5.30 个百分点，PatternNet 则下降 3.67 个百分点。
