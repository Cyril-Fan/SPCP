# Development checks / 开发检查

Run these commands from the repository root:

```bash
python scripts/prepare_protocol_assets.py
python scripts/check_release.py
python -m unittest discover -s tests
```

- `prepare_protocol_assets.py` extracts the bundled splits and image-hash ledgers after checking the archive and member hashes.
- `check_release.py` checks the existing source inventory and validates the full-method configurations for seeds 1, 2 and 3 without loading a model.
- The unit tests use synthetic images and a small CLIP-compatible model to exercise the objective, frozen parameters, graph construction, checkpoint recovery and transfer evaluation.

[GitHub Actions](https://github.com/Cyril-Fan/SPCP/actions/workflows/checks.yml) runs these checks in the pinned Python 3.8.20 CPU environment. Benchmark accuracy requires the datasets and GPU training described in the root README.

The existing `docs/source_identity.json` records hashes for runtime modules, setup scripts, tests and configurations. When editing a listed file, update its hash after reviewing the change. Checkpoint compatibility also depends on the runtime source files; see the [checkpoint guide](REPRODUCIBILITY.md#checkpoints).

在仓库根目录依次执行上述命令，检查协议文件、源码清单、配置及合成数据测试。CI 使用固定的 Python 3.8.20 CPU 环境；真实数据集上的准确率需按 README 运行训练和评估。修改清单覆盖的文件后，需要同步更新现有哈希；检查点兼容条件见复现说明。
