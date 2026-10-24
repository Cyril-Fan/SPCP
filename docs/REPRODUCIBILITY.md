# Full SPCP reproducibility / 完整 SPCP 复现说明

## Source mapping / 源码对应

| Concept / 概念 | File / 文件 |
|---|---|
| Frozen CLIP and cross-modal prompt interaction X | `spcp/base_model.py` |
| Individual text anchors A and same-input image alignment I | `spcp/models/common.py` |
| Fixed source-class neighbor graph | `spcp/models/source_graph.py` |
| Full coordinate differences E; `CE + I + 0.5 A + 0.5 E` | `spcp/models/spcp.py` |
| Training, evaluation and checkpoint restoration | `training.py`, `evaluation.py`, `checkpoint.py` in `spcp/` |
| Frozen data/CLIP contracts | `spcp/config.py` |
| Public CLI / 对外入口 | `python -m spcp` |

SPCP jointly integrates text semantic preservation S, image alignment I, and prompt interaction X. S combines individual class anchoring A with full coordinate differences E. All model and runtime interfaces use the SPCP name. The source inventory in `source_identity.json` lists SHA-256 hashes for runtime modules, setup scripts, tests and configurations.

S/I/X 共同组成完整方法，S 包含逐类锚点 A 与完整坐标差 E。模型与运行接口统一使用 SPCP。固定描述、类别映射、图像身份和结果数值均属于复现协议。

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

Full-method per-seed results and summaries are in `results/`. The setting η=0.50 was selected through target-informed exploratory DG screening, documented in `assets/admissions/spcp_eta050.json`. The reported DG result therefore does not independently validate source-only hyperparameter selection. The command-line interface supports full-method training and evaluation for B2N, CD and DG.

HM 先按数据集和 seed 计算再汇总，CD/DG 的目标域均值不包含源域。`results/` 提供完整方法的逐 seed 结果及汇总。η=0.50 的选择使用了探索性的 DG 目标域结果，因此报告的 DG 成绩不构成独立的源域超参数选择验证。

## Checkpoints and data / 检查点与数据

Preserve the whole output directory when backing up a run, including `metadata.json`, checkpoint and completion/audit records. Replays and resume operations enforce the same configuration, source tree, seed and data identities. Original archive checkpoints may be rejected after a source-tree change even when the mathematical kernels are unchanged. This code release is intended for fresh reproducible runs and does not remove those guards.

请保留完整运行目录，恢复时保持代码树、配置、seed、绑定与数据身份一致。历史归档检查点可能因源树身份不同而被拒绝，不应删除校验强行加载。下载、绑定及训练命令见根目录 README。

The exact CLIP source matches the CoOp-distributed implementation at `ff61507c790454bce7c5052c3ac39e60772f1f89`; it is not interchangeable with the latest upstream CLIP package under these guards. APPLeNet split/release identities are pinned to `42bc4654311da9773b59979dc71fe9f60534ab84`.

CLIP 和数据下载由上游托管；本仓库提供固定版本及哈希。数据集图像、下载的骨干权重与本机绑定不提交到 Git。
