"""Fixed source-class neighbor graph built from frozen semantic features."""
import copy
import torch
from .common import MethodModel
from ..checkpoint import state_hash


def neighbors(q, k=3, exclude_duplicates=False):
    q = q.detach().float().cpu()
    count = len(q)
    indices = torch.zeros(count, min(k, max(0, count-1)), dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    with torch.autocast(device_type="cpu", enabled=False):
        scores = q @ q.T
    for i in range(count):
        candidates = [j for j in range(count) if j != i and
                      (not exclude_duplicates or float((q[i]-q[j]).norm()) >= 1e-6)]
        selected = sorted(candidates, key=lambda j: (-float(scores[i, j]), j))[:indices.shape[1]]
        if selected:
            indices[i, :len(selected)] = torch.tensor(selected)
            mask[i, :len(selected)] = True
    return indices, mask


class SourceGraphModel(MethodModel):
    exclude_duplicates = False

    def __init__(self, clip_model, clip_module, config):
        super().__init__(clip_model, clip_module, config)
        self.register_buffer("source_q", torch.empty(0, self.embedding_dim, device=self.device))
        self.register_buffer("source_neighbors", torch.empty(0, 0, dtype=torch.long, device=self.device))
        self.register_buffer("source_neighbor_mask", torch.empty(0, 0, dtype=torch.bool, device=self.device))
        self.source_classnames = ()
        self.source_templates = ()
        self._source_seed = config.get("_seed", 1)

    def graph_tensors(self, q):
        index, mask = neighbors(q, self.method_config["neighbors"], self.exclude_duplicates)
        return {"source_q": q.detach().float().cpu().clone(), "source_neighbors": index,
                "source_neighbor_mask": mask}

    def _method_buffers(self):
        return {name: getattr(self, name) for name in ("source_q", "source_neighbors", "source_neighbor_mask")}

    @torch.no_grad()
    def prepare(self, bundle, eval_preprocess, seed, output_dir):
        if bundle.metadata.get("role") != "source" or not bundle.train:
            raise ValueError("Source graph requires training source supports")
        if self.source_classnames:
            raise ValueError("Source graph is already sealed; restore it instead of recalibrating")
        self._require_vocabulary()
        self.source_classnames, self.source_templates = self.classnames, self.taxonomy_templates
        for name, value in self.graph_tensors(self.reference_text()).items():
            setattr(self, name, value.to(self.device))
        self._source_seed = seed
        self.method_receipt = dict(method=self.method_name, graph_version=1, source_seed=seed,
                                   source_classnames=list(self.source_classnames), source_templates=list(self.source_templates),
                                   support_sha256=bundle.metadata.get("support_sha256"),
                                   graph_sha256=state_hash(self._method_buffers()),
                                   extra_images=0, extra_labels=0,
                                   graph_use="source training classes only; immutable during target evaluation")

    def require_source_graph(self):
        if not self.source_classnames or self.classnames != self.source_classnames or self.taxonomy_templates != self.source_templates:
            raise ValueError("Training loss requires the saved source graph and original source class order/templates")

    def validate_method_state(self, state):
        if state.get("method") != self.method_name or state.get("version") != 1 or state.get("config") != self.method_config:
            raise ValueError("Checkpoint graph method/version/config differs")
        buffers, receipt = state.get("buffers", {}), state.get("receipt", {})
        if set(buffers) != set(self._method_buffers()) or receipt.get("graph_version") != 1:
            raise ValueError("Checkpoint source graph is incomplete/unprepared")
        names, templates = receipt.get("source_classnames", []), receipt.get("source_templates", [])
        q = buffers["source_q"]
        if not names or len(set(names)) != len(names) or len(templates) != len(names):
            raise ValueError("Invalid saved source vocabulary")
        if q.dtype != torch.float32 or q.shape != (len(names), self.embedding_dim) or not torch.isfinite(q).all():
            raise ValueError("Invalid source anchors")
        if not torch.allclose(q.norm(dim=-1), torch.ones(len(names)), atol=2e-5, rtol=2e-5):
            raise ValueError("Source anchors must be unit features")
        expected = self.graph_tensors(q)
        for name, value in expected.items():
            actual = buffers[name]
            if not isinstance(actual, torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual.cpu(), value):
                raise ValueError("Source graph edge/anchor state differs: " + name)
        if receipt.get("graph_sha256") != state_hash(buffers):
            raise ValueError("Source graph hash differs")

    @torch.no_grad()
    def load_method_state_dict(self, state):
        self.validate_method_state(state)
        for name, value in state["buffers"].items():
            setattr(self, name, value.to(self.device).clone())
        self.method_receipt = copy.deepcopy(state["receipt"])
        self.source_classnames = tuple(self.method_receipt["source_classnames"])
        self.source_templates = tuple(self.method_receipt["source_templates"])
        self._source_seed = self.method_receipt["source_seed"]
