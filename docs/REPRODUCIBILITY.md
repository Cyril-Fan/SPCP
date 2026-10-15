# Full SPCP reproducibility / 完整 SPCP 复现说明

## Source mapping / 源码对应

| Concept / 概念 | File / 文件 |
|---|---|
| Frozen CLIP and cross-modal prompt interaction X | `t2methods/base_model.py` |
| Individual text anchors A and same-input image alignment I | `t2methods/models/common.py` |
| Fixed source-class neighbor graph | `t2methods/models/source_graph.py` |
| Full coordinate differences E; `CE + I + 0.5 A + 0.5 E` | `t2methods/models/cpta.py` |
| Training, evaluation and checkpoint restoration | `training.py`, `evaluation.py`, `checkpoint.py` in `t2methods/` |
| Frozen data/CLIP contracts | `t2methods/config.py` |
| Public CLI / 对外入口 | `python -m spcp` |

SPCP jointly integrates text semantic preservation S, image alignment I, and prompt interaction X. S combines A with E. The prompt, graph and loss kernels retain the verified original source bytes; the public model registry and validator expose full SPCP only. The historical runtime ID is `CPTA`. The older base class name `T2NetModel` denotes a local independent prompt reconstruction, not official T2Net author code. Source hashes and interface changes are recorded in `source_identity.json`.

S/I/X 共同组成完整方法，S 包含 A 与 E。提示、源图及损失计算核心保持原始字节，对外模型注册及验证接口限制为完整 SPCP。内部历史类名不表示获取了 T2Net 官方源码。源码身份及接口变动另有清单。

## Fixed protocols / 固定协议

| Protocol | Source / 源域 | Evaluation / 评估 | Epochs |
|---|---|---|---:|
| B2N | Base supports in each of four original datasets | Separate Base/New vocabularies; HM | 50 |
| CD | PatternNet | RSICD, NWPU-RESISC45, MLRSNet | 10 |
| DG | PatternNetv2 | RSICDv2, NWPU-RESISC45v2, MLRSNetv2; 16 common classes | 10 |

All use 16-shot source supports, seeds 1/2/3, CLIP ViT-B/16, 16 prompt tokens, 12 prompted blocks, and a fixed final-epoch checkpoint. B2N uses ceil-half sorted integer class labels. Targets do not supply training supports. Fixed descriptions, neighbor settings, preprocessing and label maps must be retained.

所有任务使用固定划分、16-shot 源域支持集、seed 1/2/3、ViT-B/16、16 个提示 token、12 个提示层及最终 epoch 检查点。B2N 对排序后的类别取 ceil-half 作为 Base。目标域不提供训练支持样本，不能改换划分、类别映射或 v2 图像集合后仍称为同一协议。

## Aggregation and scope / 汇总及证据范围

Compute B2N HM for each dataset and seed, then average the four datasets within a seed and finally average the three seed aggregates. CD/DG target averages exclude PatternNet/PatternNetv2 source accuracy. Sample SD uses `ddof=1`. Do not reconstruct HM from already rounded mean Base/New values.

Existing full-method summaries are in `results/`; these are saved experiments, not newly executed publication tests. The historical choice η=0.50 followed exploratory DG screening informed by target results. `assets/admissions/cpta_eta050.json` preserves that provenance; it is not an independent source-only selection receipt. The reported DG result is not independent validation of a parameter-selection rule. The public release exposes only the full-method training/evaluation tasks requested by the authors.

HM 先按数据集和 seed 计算再汇总，CD/DG 的目标域均值不包含源域。`results/` 为既有完整方法结果。本方法 η=0.50 的历史选择包含探索性的目标域信息，不能表述为独立源域验证。发布检查不增加真实实验或新的显著性结论。

## Checkpoints and data / 检查点与数据

Preserve the whole output directory when backing up a run, including `metadata.json`, checkpoint and completion/audit records. Replays and resume operations enforce the same configuration, source tree, seed and data identities. Original archive checkpoints may be rejected after a source-tree change even when the mathematical kernels are unchanged. This code release is intended for fresh reproducible runs and does not remove those guards.

请保留完整运行目录，恢复时保持代码树、配置、seed、绑定与数据身份一致。历史归档检查点可能因源树身份不同而被拒绝，不应删除校验强行加载。下载、绑定及训练命令见根目录 README。

The exact CLIP source matches the CoOp-distributed implementation at `ff61507c790454bce7c5052c3ac39e60772f1f89`; it is not interchangeable with the latest upstream CLIP package under these guards. APPLeNet split/release identities are pinned to `42bc4654311da9773b59979dc71fe9f60534ab84`.

CLIP 和数据下载由上游托管；本仓库提供固定版本及哈希。数据集图像、下载的骨干权重与本机绑定不提交到 Git。
