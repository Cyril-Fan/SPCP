"""Atomic artifacts and relocatable checkpoint identities.

Metadata utilities import tensor serialization only when required."""
import hashlib
import json
import os
from pathlib import Path
import uuid


SCHEMA = "spcp.checkpoint.v1"


def absolute_path(path):
    """Lexical normalization for our managed metadata/output artifacts only."""
    return Path(os.path.abspath(str(path)))


def _is_link(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def source_hashes(root=None):
    root = Path(root or Path(__file__).parent)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}


def state_hash(value):
    """Length-delimited recursive hash of tensor state, including fixed buffers."""
    import torch
    digest = hashlib.sha256()
    def token(data):
        digest.update(str(len(data)).encode("ascii") + b":" + data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            token(b"tensor")
            token(str(tensor.dtype).encode("ascii"))
            token(str(tuple(tensor.shape)).encode("ascii"))
            token(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            token(b"dict")
            for key in sorted(item):
                token(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            token(b"sequence")
            token(str(len(item)).encode("ascii"))
            for part in item:
                visit(part)
        else:
            token(json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8"))
    visit(value)
    return digest.hexdigest()


def model_state(model):
    return {"prompts": model.prompt_state_dict(), "method": model.method_state_dict()}


def model_state_hash(model):
    return state_hash(model_state(model))


def backbone_hash(model):
    return state_hash(model.clip_model.state_dict())


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    import torch
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Unknown checkpoint schema; use checkpoints produced by this SPCP source tree")
    if value.get("state_sha256") != state_hash({"prompts": value["prompts"], "method": value["method_state"]}):
        raise ValueError("Checkpoint prompt/method state hash mismatch")
    return value


def restore_model(model, saved):
    model.validate_method_state(saved["method_state"])
    model.load_prompt_state_dict(saved["prompts"])
    model.load_method_state_dict(saved["method_state"])
    if model_state_hash(model) != saved["state_sha256"]:
        raise ValueError("Restored model state differs from checkpoint")


def capture_rng(generator, device):
    import random
    import numpy as np
    import torch
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state(), "loader": generator.get_state(), "cuda": None}
    if str(device).startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(saved, generator, device):
    import random
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"])
    generator.set_state(saved["loader"])
    if str(device).startswith("cuda"):
        if saved.get("cuda") is None:
            raise ValueError("CUDA resume requires saved CUDA RNG states")
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA resume device inventory differs")
        torch.cuda.set_rng_state_all(saved["cuda"])
    elif saved.get("cuda") is not None:
        raise ValueError("CUDA checkpoint cannot resume as a CPU trajectory")


def finish_run(output, request_identity, state_sha256):
    output = Path(output)
    artifacts = {p.relative_to(output).as_posix(): file_hash(p)
                 for p in sorted(output.iterdir())
                 if p.is_file() and (p.name in ("final.pt", "results.json", "metadata.json", "audit.json", "source_validation_pixels.json", "source_validation_risk.json",
                     "loss_contract.json", "initialization.json", "gradient_match.json", "preparation_cost.json")
                                     or p.suffix in (".csv", ".npz", ".jsonl"))}
    if not {"final.pt", "results.json", "metadata.json"}.issubset(artifacts):
        raise ValueError("Cannot mark an incomplete fit as complete")
    marker = {"schema": SCHEMA, "request_identity": request_identity,
              "state_sha256": state_sha256, "artifacts": artifacts}
    write_json(output / "complete.json", marker)
    return marker


def completed_run(output, request_identity):
    """A status string alone never makes an existing fit reusable."""
    output = absolute_path(output)
    try:
        if _is_link(output) or _is_link(output / "complete.json"):
            return False
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker.get("schema") != SCHEMA or marker.get("request_identity") != request_identity:
            return False
        artifacts = marker["artifacts"]
        if not {"final.pt", "results.json", "metadata.json"}.issubset(artifacts):
            return False
        for name, expected in artifacts.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name or ".." in name or Path(name).is_absolute():
                return False
            path = absolute_path(output / name)
            if path.parent != output or _is_link(path) or not path.is_file() or file_hash(path) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _recovery_status(output):
    """Inspect recovery eligibility without changing successful evidence."""
    output = absolute_path(output)
    if not output.is_dir() or _is_link(output):
        raise ValueError("Recovery requires an existing managed output directory, never a link/junction")
    status = {}
    status_path = output / "status.json"
    if _is_link(status_path):
        raise ValueError("Recovery status must not be a link/junction")
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    if (output / "complete.json").exists() or _is_link(output / "complete.json") or status.get("status") in ("completed", "evaluation_complete"):
        raise FileExistsError("Successful or hash-mismatched completed artifacts require explicit investigation, never failed retry")
    for name in ("final.pt", "last.pt", "failure.json"):
        if _is_link(output / name):
            raise ValueError("Recovery artifact must not be a link/junction: " + name)
    return status


def validate_epoch_resume(output, checkpoint):
    """An explicit epoch resume also cannot bypass successful-output guards."""
    output, checkpoint = absolute_path(output), absolute_path(checkpoint)
    _recovery_status(output)
    if (output / "final.pt").exists():
        raise FileExistsError("A committed final checkpoint requires final recovery, never epoch resume")
    if checkpoint != output / "last.pt" or not checkpoint.is_file() or _is_link(checkpoint):
        raise ValueError("Epoch resume requires this output's regular last.pt checkpoint")


def failed_retry_kind(output):
    """Never treat a corrupt/mismatched success marker as permission to redo."""
    output = absolute_path(output)
    status = _recovery_status(output)
    if not (output / "failure.json").is_file() and status.get("status") not in ("failed", "runtime_limit"):
        raise ValueError("Failed retry requires a recorded failure; interrupted epoch continuation uses --resume")
    if (output / "final.pt").is_file():
        return "evaluate_final"
    if (output / "last.pt").is_file():
        return "resume_epoch"
    return "restart_pre_checkpoint"


def interrupted_retry_kind(output):
    """Recover an interrupted run only from an existing matching checkpoint."""
    output = absolute_path(output)
    status = _recovery_status(output)
    if (output / "failure.json").exists() or status.get("status") in ("failed", "runtime_limit"):
        raise ValueError("Recorded failures use --retry-failed, not --recover-interrupted")
    if (output / "final.pt").is_file():
        return "evaluate_final"
    if (output / "last.pt").is_file():
        return "resume_epoch"
    raise ValueError("Interruption recovery requires a committed final.pt or last.pt; never restart an unknown output")


def record_interruption(output, kind, saved):
    """Call only after Trainer validates the committed checkpoint provenance."""
    output = absolute_path(output)
    if interrupted_retry_kind(output) != kind:
        raise ValueError("Interrupted recovery state changed before recording")
    checkpoint = output / ("final.pt" if kind == "evaluate_final" else "last.pt")
    receipt = {"type": "InterruptedRun", "retry_kind": kind, "checkpoint": checkpoint.name,
               "checkpoint_sha256": file_hash(checkpoint), "completed_epochs": saved["completed_epochs"],
               "request_identity": saved["metadata"]["request_identity"], "provenance_validated": True,
               "measured_success": False, "message": "Recovery of an unrecorded interruption from an existing checkpoint"}
    write_json(output / "interruption.json", receipt)
    write_json(output / "failure.json", receipt)
    write_json(output / "status.json", {"status": "failed", "error_type": "InterruptedRun", "retry_kind": kind})
    return receipt


def archive_failed_attempt(output, kind):
    """Archive known flat failed artifacts, preserving all committed checkpoints."""
    output = Path(os.path.abspath(str(output)))
    if _is_link(output):
        raise ValueError("Failure output must not be a link/junction")
    if failed_retry_kind(output) != kind:
        raise ValueError("Failed retry state changed before archival")
    archive = output / "attempts" / ("failed_" + uuid.uuid4().hex)
    if os.path.commonpath([str(output), str(archive)]) != str(output):
        raise ValueError("Failure archive escapes the managed output")
    archive.mkdir(parents=True, exist_ok=False)
    names = {"failure.json", "interruption.json", "status.json", "results.json", "audit.json"}
    if kind != "evaluate_final":
        names.update({"metadata.json", "history.jsonl", "calibration.json", "audit.json"})
    moved = []
    for path in output.iterdir():
        if not path.is_file() or path.name not in names and path.suffix not in (".npz", ".csv", ".jsonl", ".partial") and not path.name.startswith("source_validation_"):
            continue
        if _is_link(path):
            raise ValueError("Refuse to archive a symbolic-link artifact")
        destination = archive / path.name
        if os.path.commonpath([str(output), str(destination)]) != str(output):
            raise ValueError("Failure artifact escapes managed output")
        path.replace(destination)
        moved.append(path.name)
    write_json(archive / "attempt.json", {"retry_kind": kind, "archived_files": sorted(moved),
                                          "committed_checkpoints_moved": False})
    return str(archive)
