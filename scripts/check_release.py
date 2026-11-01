"""Check Python syntax and B2N/CD/DG configurations without loading a model."""

import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from spcp.config import load_config, validate_config


def main():
    sources = [p for folder in ("spcp", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    if len(configs) != 6:
        raise ValueError("Expected four B2N, one CD and one DG configurations")
    checks = [validate_config(load_config(p, s)) for p in configs for s in (1, 2, 3)]
    print(
        json.dumps(
            {
                "pass": True,
                "python_files": len(sources),
                "full_configs": len(configs),
                "config_seed_checks": len(checks),
                "torch_loaded": "torch" in sys.modules,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
