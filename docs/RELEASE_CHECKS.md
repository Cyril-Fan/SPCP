# Release checks / 发布检查

Date / 日期：2026-10-16. Scope / 范围：SPCP, B2N, CD and DG.

- Six full-method configurations pass validation for seeds 1, 2 and 3: **18 metadata checks**, without loading PyTorch or CLIP.
- **Nine synthetic CPU tests** cover the joint objective, frozen encoders, same-input image alignment, fixed graph integrity, optimizer/checkpoint resume, DG seed-one training, replay and transfer evaluation.
- A before/after numerical comparison verifies that **16 forward-output, loss and trainable-gradient tensors are bitwise identical** after the package and method naming cleanup.
- The release source inventory is recorded in `source_identity.json` and checked by `scripts/check_release.py`.
- Taxonomy class entries, frozen description strings, split bytes, image ledgers and numeric result fields are unchanged. Descriptive metadata and runtime names have been updated; checkpoints remain tied to their source/configuration identity.
- Protocol metadata restoration verifies the archive inventory, CRC, byte lengths and SHA-256 hashes. Images, weights and local bindings are excluded from Git.
- Original SPCP code uses the MIT License. Third-party copyright and license notices are retained.
- Checks run locally with Python 3.8.20, PyTorch 2.4.1+cpu and torchvision 0.19.1+cpu. Dataset downloads and full benchmark training are outside these release checks; no GPU experiment was run.

已完成 18 项配置检查、9 项 CPU 测试及 16 个张量的修改前后逐位一致性核验。固定描述、类别映射、划分、图像 ledger 和实验数值保留。发布检查不替代真实数据集训练。

```bash
python scripts/prepare_protocol_assets.py
python scripts/check_release.py
python -m unittest discover -s tests
```
