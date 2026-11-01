# Development checks / 开发检查

Run from the repository root:

```bash
python scripts/prepare_protocol_assets.py
python scripts/check_release.py
python -m unittest discover -s tests
```

- `prepare_protocol_assets.py` verifies and extracts the bundled splits and image-hash ledgers.
- `check_release.py` checks Python syntax and the six full-method configurations for seeds 1, 2 and 3, without loading a model. It does not compare source-file hashes or enforce a formatter.
- Tests cover losses and gradients, frozen parameters, source graphs, checkpoint compatibility and recovery, transfer evaluation, and published result aggregation.

[GitHub Actions](https://github.com/Cyril-Fan/SPCP/actions/workflows/checks.yml) uses Python 3.8.20 and the pinned core packages with CPU PyTorch. Synthetic tests check functionality; benchmark accuracy requires dataset images and GPU training.

在仓库根目录运行上述命令，检查协议资产、配置及合成数据流程。源码注释和格式无需更新哈希清单；CI 不设源码字节或格式门槛。测试覆盖损失/梯度、冻结参数、源图、检查点兼容与恢复、迁移评估及既有结果汇总。CPU 功能检查不代表重新复现论文准确率。
