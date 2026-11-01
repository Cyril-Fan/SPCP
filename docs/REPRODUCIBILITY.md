# Reproducing SPCP / SPCP 复现说明

## Code map / 代码对应

| Component | Implementation |
|---|---|
| Frozen CLIP and cross-modal prompt interaction X | `spcp/base_model.py` |
| Individual text anchors A and same-input image alignment I | `spcp/models/common.py` |
| Fixed source-class neighbor graph | `spcp/models/source_graph.py` |
| Full class-difference vectors E and the SPCP objective | `spcp/models/spcp.py` |
| Training, evaluation and checkpoints | `spcp/training.py`, `spcp/evaluation.py`, `spcp/checkpoint.py` |
| Configuration and asset checks | `spcp/config.py` |
| Command-line entry point | `python -m spcp` |

## Environment and configurations / 环境与配置

Use the pinned experiment environment and installation commands in the root README. The CLIP implementation is the CoOp-distributed source at `ff61507c790454bce7c5052c3ac39e60772f1f89`; dataset releases and splits use the APPLeNet records at `42bc4654311da9773b59979dc71fe9f60534ab84`.

The public CLI runs the full SPCP method with the published settings. Configuration validation checks the backbone, prompt geometry, objective coefficients, training recipe, seeds and dataset identities. Changing these fields produces a validation error; the CLI reproduces the provided full-method configurations.

实验环境及安装命令见根目录 README。公开命令行入口使用论文完整方法和固定配置；骨干、提示结构、损失系数、训练设置、seed 或数据身份与配置不符时，校验会报告错误。

## Evaluation protocols / 评估协议

| Protocol | Source | Evaluation | Epochs |
|---|---|---|---:|
| B2N | Base supports in each of four original datasets | Separate Base/New vocabularies; HM | 50 |
| CD | PatternNet | RSICD, NWPU-RESISC45, MLRSNet | 10 |
| DG | PatternNetv2 | RSICDv2, NWPU-RESISC45v2, MLRSNetv2; 16 common classes | 10 |

All configurations use 16-shot source supports, seeds 1/2/3, CLIP ViT-B/16, 16 prompt tokens, 12 prompted blocks and the final-epoch checkpoint. B2N assigns the first half, rounded up, of sorted integer class labels to Base. Target datasets provide evaluation images only. DG uses the released v2 image collections and their fixed shared-class mapping.

各协议均使用 16-shot 源域支持集、seed 1/2/3、ViT-B/16、16 个提示 token、12 个提示层及最终 epoch 检查点。B2N 按排序后的整数标签取前一半并向上取整为 Base；目标域仅用于评估。DG 使用 v2 图像发布集合及固定共有类别映射。

## Implementation details / 实现细节

The first four shallow text context tokens use the embeddings of “a photo of a”; remaining context tokens and deeper source prompts use Normal(0, 0.02). Two projections are shared within the respective layer groups. Prompted and frozen image features use the same augmented input. Frozen text references use the fixed hierarchy sentences, prepared before inspecting experiment scores. Readable class-name mappings are stored in the taxonomy JSON; integer labels and support IDs are unchanged.

浅层首组文本提示的前四个 token 使用 “a photo of a” 的词嵌入，其余 token 和深层源提示按 Normal(0, 0.02) 初始化；两个投影分别在各自层组内共享。图像对齐的两个分支使用同一增强输入。冻结文本参考使用查看实验结果前准备的固定层级描述；可读类名映射保存在 taxonomy JSON 中，整数标签及支持集 ID 不变。

## Result aggregation / 结果汇总

Compute B2N HM separately for each dataset and seed, average the four datasets within each seed, then summarize the three seed aggregates. CD/DG averages exclude source accuracy. Sample SD uses `ddof=1`. The `results/` directory contains full-method per-seed results and summaries; calculate aggregates from unrounded values.

HM 按数据集和 seed 分别计算，再在每个 seed 内对四个数据集取平均，最后汇总三个 seed。CD/DG 的均值不包含源域，样本 SD 使用 `ddof=1`。逐 seed 结果及汇总位于 `results/`，计算时使用未舍入数值。

## Checkpoints

Keep each run directory together. `last.pt` resumes training at a completed epoch, including optimizer, AMP scaler and random states; `final.pt` supports evaluation. Matching execution settings, seed, source supports/class mapping, templates, tokenizer and CLIP weights are required. Prompt names/shapes, fixed graph buffers and artifact integrity are checked. Comments, formatting, local paths and descriptive metadata do not determine checkpoint compatibility. Older checkpoints use the same checks on their saved configuration; no conversion is needed.

保留完整运行目录。`last.pt` 恢复最后完成的 epoch、优化器、AMP scaler 和随机状态；`final.pt` 用于评估。实际配置、seed、支持集与类别映射、模板、tokenizer 和 CLIP 权重必须匹配，并检查提示参数名称/形状、固定源图及输出完整性。注释、格式、本机路径和说明性元数据不影响兼容性；旧检查点直接按其中保存的配置核对，无需转换。

## Metadata fields / 元数据字段

Published configs specify `data.split` explicitly. Direct loader calls for the four original datasets may omit it to use `assets/raw/splits/`; prepare these files with `python scripts/prepare_protocol_assets.py`. DG and custom datasets require an explicit split path.

公开配置均显式指定 `data.split`。直接调用加载器时，四个原始数据集可省略此字段，使用 `assets/raw/splits/` 中的随附划分；先运行 `python scripts/prepare_protocol_assets.py` 准备文件。DG 和自定义数据集须明确指定划分路径。

- `split_sha256` and `split_commit` identify the split file and its pinned upstream revision.

Bundled image ledgers use relative image IDs; local paths are resolved from the asset binding. The preparation receipt preserves image-verification provenance, and its ledger hashes identify the portable public copies.

随附图像清单使用相对图像 ID，本机路径由资产绑定解析。准备记录保留图像核验来源，其中的清单哈希对应公开的可移植副本。
