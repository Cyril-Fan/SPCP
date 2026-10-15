# Release checks / 发布检查

Date / 日期：2026-10-16. Scope / 范围：full SPCP, B2N, CD and DG.

- Six full-method configurations checked for seeds 1/2/3: **18 metadata checks passed**, without loading a model.
- **Eight synthetic CPU tests passed**, covering the joint objective, frozen parameters, source graph, same-input teacher, checkpoint integrity, optimizer resume, and transfer evaluation.
- **20 original source files** match their retained SHA-256 identities. Public interface changes restrict registration/validation to full SPCP and add the `spcp` entry point; prompt/loss kernels are unchanged.
- Protocol metadata are losslessly bundled and verified against per-file hashes when restored. Dataset images, local asset bindings and downloaded weights are excluded from Git.
- Documentation provides English and Chinese instructions, provider download links, the pinned CoOp-distributed CLIP source, and the official OpenAI weight hash. Command-line network downloads were not completed in the release-check environment because DNS resolution was unavailable; the downloader's URLs are pinned, and actual file hashes remain mandatory.
- No new benchmark training and no paid GPU session were used for this release.

已完成六份完整方法配置的 18 项 seed 检查、8 项人工数据 CPU 测试及 20 份原始源码身份核对。中英文文档包括下载、资产绑定、训练、评估和恢复命令。发布环境的命令行 DNS 无法解析外网，因此未声称已完成实际权重/数据下载；固定地址及哈希检查继续保留。发布检查不替代真实基准实验。

Run / 复查命令：

```bash
python scripts/prepare_protocol_assets.py
python scripts/check_release.py
python -m unittest discover -s tests
```
