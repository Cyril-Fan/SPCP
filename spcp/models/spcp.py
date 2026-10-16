"""SPCP reconstructs directed coordinate differences, including their lengths."""
import torch
from .source_graph import SourceGraphModel
from ..settings import settings


def edge_loss(text, q, index, mask, reference_delta, denominator, control="difference"):
    with torch.autocast(device_type=text.device.type, enabled=False):
        t, q = text.float(), q.detach().float()
        delta = t[:, None] - t[index]
        if control == "gram":
            values = ((t[:, None]*t[index]).sum(-1) - (q[:, None]*q[index]).sum(-1)).square()
        else:
            values = (delta - reference_delta.detach()).square().sum(-1)
            if control == "difference":
                values = values / denominator.detach()
            elif control != "unnormalized":
                raise ValueError("Unknown SPCP edge objective")
        count = mask.sum(-1)
        node = (values * mask).sum(-1) / count.clamp_min(1)
        fallback = 1 - (t*q).sum(-1)
        node = torch.where(count > 0, node, fallback)
        return node.mean(), node, delta


class SPCPModel(SourceGraphModel):
    method_name = "SPCP"
    exclude_duplicates = True

    def __init__(self, clip_model, clip_module, config):
        super().__init__(clip_model, clip_module, config)
        self.method_config = settings(self.method_config)
        self.register_buffer("source_delta", torch.empty(0, 0, self.embedding_dim, device=self.device))
        self.register_buffer("source_denominator", torch.empty(0, 0, device=self.device))
        self._last_diagnostics = {}

    def _method_buffers(self):
        return {**super()._method_buffers(), "source_delta": self.source_delta,
                "source_denominator": self.source_denominator}

    def graph_tensors(self, q):
        values = super().graph_tensors(q)
        q, index = values["source_q"], values["source_neighbors"]
        delta = q[:, None] - q[index]
        values.update(source_delta=delta, source_denominator=delta.square().sum(-1) + self.method_config["geometry_epsilon"]**2)
        return values

    def loss_terms(self, images, labels, ids=None, epoch=0):
        terms = super().loss_terms(images, labels, ids, epoch)
        if not self.enabled or self.method_config["eta"] == 0:
            return terms
        self.require_source_graph()
        t = terms["output"]["text_features"]
        edge, node, delta = edge_loss(t, self.source_q, self.source_neighbors, self.source_neighbor_mask,
                                    self.source_delta, self.source_denominator, self.method_config["control"])
        eta = self.method_config["eta"]
        text_mix = (1-eta)*terms["text_alignment"] + eta*edge
        terms["mechanism_loss"] = edge
        terms["diagnostic_base_loss"] = terms["ce"]
        terms["diagnostic_aux_loss"] = text_mix
        terms["loss"] = terms["ce"] + self.alignment_weight * (
            self.text_alignment_weight*text_mix + self.image_alignment_weight*terms["image_alignment"])
        with torch.no_grad():
            mask = self.source_neighbor_mask
            lengths = self.source_delta.norm(dim=-1)
            student_lengths = delta.detach().norm(dim=-1)
            direction = 1 - torch.nn.functional.cosine_similarity(delta.detach(), self.source_delta, dim=-1)
            self._last_diagnostics = dict(edge_loss=float(edge.detach()), per_class_edge_loss=node.detach().tolist(),
                source_classnames=list(self.source_classnames), teacher_edge_lengths=lengths[mask].tolist(),
                student_edge_length_ratio=(student_lengths/lengths.clamp_min(1e-6))[mask].tolist(),
                direction_error=direction[mask].tolist(), isolated_nodes=int((~mask.any(-1)).sum()),
                text_alignment=float(terms["text_alignment"].detach()), control=self.method_config["control"])
        return terms

    def mechanism_diagnostics(self):
        return {"method": "SPCP", "enabled": self.enabled, **self._last_diagnostics}
