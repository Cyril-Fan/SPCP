"""Frozen original CLIP with explicitly declared T2Net prompt reconstruction.

The paper does not supply TKI tensor equations. This implementation transfers
static source prompt tokens with two shared projections; see model_logic.md.
It contains no NSCP, adapter, hierarchy classifier, or logit correction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class T2NetModel(nn.Module):
    """Prompted students and unchanged CLIP teachers with dynamic vocabulary."""

    def __init__(
        self, clip_model: nn.Module, clip_module: Any, config: dict[str, Any]
    ) -> None:
        super().__init__()
        settings = config.get("model", config)
        if not isinstance(settings, Mapping):
            raise TypeError("config.model must be a mapping")
        self.prompt_length = self._integer(settings.get("prompt_length", 16), "prompt_length")
        self.depth = self._integer(settings.get("depth", 12), "depth")
        self.shallow_layers = self._integer(settings.get("shallow_layers", 6), "shallow_layers")
        self.context_init = str(settings.get("context_init", "a photo of a"))
        self.init_std = float(settings.get("init_std", 0.02))
        self.interaction = str(settings.get("interaction", "directional_shared_projection"))
        if self.interaction not in {"directional_shared_projection", "none"}:
            raise ValueError("interaction must be directional_shared_projection or none")
        if not 0 < self.shallow_layers < self.depth:
            raise ValueError("shallow_layers must lie strictly between zero and depth")
        if not 0 < self.init_std < float("inf"):
            raise ValueError("init_std must be positive and finite")

        self.clip_model = clip_model
        self.clip_module = clip_module
        self._validate_backbone()
        self.context_length = int(clip_model.positional_embedding.shape[0])
        if self.prompt_length > self.context_length - 3:
            raise ValueError("prompt_length leaves no room for SOS, class, and EOT")
        self.text_width = int(clip_model.token_embedding.weight.shape[1])
        self.visual_width = int(clip_model.visual.class_embedding.numel())
        self.embedding_dim = int(clip_model.text_projection.shape[1])
        self.original_visual_tokens = int(clip_model.visual.positional_embedding.shape[0])
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)
        clip_model.eval()

        if self.interaction == "directional_shared_projection":
            self.shallow_text_prompts = self._prompt_bank(self.shallow_layers, self.text_width)
            self.deep_visual_prompts = self._prompt_bank(self.depth - self.shallow_layers, self.visual_width)
            self.text_to_visual = nn.Linear(self.text_width, self.visual_width)
            self.visual_to_text = nn.Linear(self.visual_width, self.text_width)
            self.text_to_visual.to(self.device)
            self.visual_to_text.to(self.device)
            self._initialize_phrase(self.shallow_text_prompts[0])
        else:
            self.text_prompts = self._prompt_bank(self.depth, self.text_width)
            self.visual_prompts = self._prompt_bank(self.depth, self.visual_width)
            self._initialize_phrase(self.text_prompts[0])

        self.classnames: tuple[str, ...] = ()
        self.taxonomy_templates: tuple[str, ...] = ()
        self.register_buffer("_student_tokens", torch.empty(0, self.context_length, dtype=torch.long, device=self.device), persistent=False)
        self.register_buffer("_student_fixed_embedding", torch.empty(0, self.context_length, self.text_width, device=self.device), persistent=False)
        self.register_buffer("_reference_text_features", torch.empty(0, self.embedding_dim, device=self.device), persistent=False)

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _validate_backbone(self) -> None:
        required_text = ("transformer", "token_embedding", "positional_embedding", "ln_final", "text_projection", "logit_scale", "encode_text", "encode_image")
        required_visual = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post", "proj")
        if any(not hasattr(self.clip_model, item) for item in required_text):
            raise TypeError("clip_model must implement the original OpenAI CLIP API")
        visual = getattr(self.clip_model, "visual", None)
        if visual is None or any(not hasattr(visual, item) for item in required_visual):
            raise TypeError("T2NetModel requires original CLIP VisionTransformer, not ResNet")
        text_blocks = getattr(self.clip_model.transformer, "resblocks", None)
        visual_blocks = getattr(visual.transformer, "resblocks", None)
        if text_blocks is None or visual_blocks is None:
            raise TypeError("original CLIP transformer.resblocks is required")
        if len(text_blocks) != self.depth or len(visual_blocks) != self.depth:
            raise ValueError("configured depth must equal both original CLIP encoder depths")
        if int(visual.conv1.in_channels) != 3:
            raise ValueError("the unchanged CLIP patch embedding must accept three channels")
        if not callable(getattr(self.clip_module, "tokenize", None)):
            raise TypeError("clip_module.tokenize is required")

    @property
    def device(self) -> torch.device:
        return self.clip_model.token_embedding.weight.device

    def _prompt_bank(self, layers: int, width: int) -> nn.Parameter:
        bank = torch.empty(layers, self.prompt_length, width, device=self.device, dtype=torch.float32)
        nn.init.normal_(bank, std=self.init_std)
        return nn.Parameter(bank)

    @torch.no_grad()
    def _initialize_phrase(self, context: Tensor) -> None:
        if not self.context_init.strip():
            return
        tokens = self._tokenize([self.context_init])
        end = int(tokens[0].argmax().item())
        count = min(end - 1, self.prompt_length)
        if count > 0:
            embedding = self.clip_model.token_embedding(tokens)[0, 1 : 1 + count]
            context[:count].copy_(embedding.to(dtype=context.dtype))

    def _tokenize(self, texts: Sequence[str]) -> Tensor:
        tokens = self.clip_module.tokenize(list(texts), context_length=self.context_length, truncate=False)
        if not isinstance(tokens, Tensor) or tokens.shape != (len(texts), self.context_length):
            raise ValueError("CLIP tokenizer returned an unexpected token tensor")
        return tokens.to(device=self.device, dtype=torch.long)

    def train(self, mode: bool = True) -> "T2NetModel":
        super().train(mode)
        self.clip_model.eval()
        return self

    @torch.no_grad()
    def set_classnames(
        self,
        classnames: Sequence[str],
        taxonomy_templates: Sequence[str] | Mapping[str, str],
    ) -> None:
        """Set ordered runtime vocabulary without changing any trainable state."""
        if isinstance(classnames, str):
            raise TypeError("classnames must be an ordered sequence, not one string")
        names = tuple(classnames)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("classnames must contain nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("classnames must be unique")
        if isinstance(taxonomy_templates, Mapping):
            missing = [name for name in names if name not in taxonomy_templates]
            if missing:
                raise ValueError(f"taxonomy is missing classnames: {missing}")
            templates = tuple(taxonomy_templates[name] for name in names)
        elif isinstance(taxonomy_templates, str):
            raise TypeError("taxonomy_templates must be sentences or a mapping")
        else:
            templates = tuple(taxonomy_templates)
        if len(templates) != len(names) or any(not isinstance(text, str) or not text.strip() for text in templates):
            raise ValueError("one nonempty fixed taxonomy sentence is required per class")
        prefix = " ".join(["X"] * self.prompt_length)
        student_sentences = [f"{prefix} {name.replace('_', ' ')}." for name in names]
        student_tokens = self._tokenize(student_sentences)
        teacher_tokens = self._tokenize(templates)
        # OpenAI CLIP keeps token_embedding in fp32 even when attention/projection
        # weights are fp16. Its text encoder casts embeddings to visual.dtype.
        embeddings = self.clip_model.token_embedding(student_tokens).to(
            dtype=self.clip_model.visual.conv1.weight.dtype
        ).detach()
        reference = self._normalize(self.clip_model.encode_text(teacher_tokens)).detach()
        if reference.shape != (len(names), self.embedding_dim) or not torch.isfinite(reference).all():
            raise ValueError("frozen taxonomy teacher returned invalid features")
        # Assignment occurs only after all validation/encoding succeeds.
        self.classnames = names
        self.taxonomy_templates = templates
        self._student_tokens = student_tokens
        self._student_fixed_embedding = embeddings
        self._reference_text_features = reference

    def _require_vocabulary(self) -> None:
        if not self.classnames:
            raise RuntimeError("set_classnames must be called before text encoding")

    @staticmethod
    def _normalize(features: Tensor) -> Tensor:
        return F.normalize(features.float(), dim=-1)

    def _text_prompt(self, layer: int) -> Tensor:
        if self.interaction == "none":
            return self.text_prompts[layer]
        if layer < self.shallow_layers:
            return self.shallow_text_prompts[layer]
        return self.visual_to_text(self.deep_visual_prompts[layer - self.shallow_layers])

    def _visual_prompt(self, layer: int) -> Tensor:
        if self.interaction == "none":
            return self.visual_prompts[layer]
        if layer < self.shallow_layers:
            return self.text_to_visual(self.shallow_text_prompts[layer])
        return self.deep_visual_prompts[layer - self.shallow_layers]

    def text_features(self) -> Tensor:
        """Return normalized prompted class features with student autograd."""
        self._require_vocabulary()
        fixed = self._student_fixed_embedding
        prompt = self._text_prompt(0).to(dtype=fixed.dtype).unsqueeze(0).expand(len(self.classnames), -1, -1)
        x = torch.cat((fixed[:, :1], prompt, fixed[:, 1 + self.prompt_length :]), dim=1)
        x = x + self.clip_model.positional_embedding.to(dtype=x.dtype)
        x = x.permute(1, 0, 2)
        for layer, block in enumerate(self.clip_model.transformer.resblocks):
            if layer:
                prompt = self._text_prompt(layer).to(dtype=x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat((x[:1], prompt, x[1 + self.prompt_length :]), dim=0)
            x = block(x)
        x = self.clip_model.ln_final(x.permute(1, 0, 2))
        end_positions = self._student_tokens.argmax(dim=-1)
        rows = torch.arange(x.shape[0], device=x.device)
        features = x[rows, end_positions] @ self.clip_model.text_projection.to(dtype=x.dtype)
        return self._normalize(features)

    def _prepare_images(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
            raise ValueError("images must be a nonempty Bx3xHxW tensor")
        return images.to(device=self.device, dtype=self.clip_model.visual.conv1.weight.dtype)

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode ordinary images with appended/replaced visual prompt slots."""
        visual = self.clip_model.visual
        x = visual.conv1(self._prepare_images(images))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(dtype=x.dtype).reshape(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if x.shape[1] != self.original_visual_tokens:
            raise ValueError("image patch count differs from unchanged CLIP positional embeddings")
        x = x + visual.positional_embedding.to(dtype=x.dtype)
        prompt = self._visual_prompt(0).to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = visual.ln_pre(torch.cat((x, prompt), dim=1)).permute(1, 0, 2)
        for layer, block in enumerate(visual.transformer.resblocks):
            if layer:
                prompt = self._visual_prompt(layer).to(dtype=x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat((x[: self.original_visual_tokens], prompt), dim=0)
            x = block(x)
        features = visual.ln_post(x[0])
        if visual.proj is not None:
            features = features @ visual.proj.to(dtype=features.dtype)
        return self._normalize(features)

    @torch.no_grad()
    def reference_image(self, images: Tensor) -> Tensor:
        """Compute the unchanged image teacher only when explicitly requested."""
        return self._normalize(self.clip_model.encode_image(self._prepare_images(images))).detach()

    def reference_text(self) -> Tensor:
        self._require_vocabulary()
        return self._reference_text_features.detach()

    def forward(self, images: Tensor, text_features: Tensor | None = None) -> dict[str, Tensor]:
        self._require_vocabulary()
        text = self.text_features() if text_features is None else text_features
        if text.ndim != 2 or text.shape != (len(self.classnames), self.embedding_dim):
            raise ValueError("cached text features must match the active class vocabulary")
        image = self.encode_image(images)
        text = self._normalize(text.to(device=image.device))
        logits = self.clip_model.logit_scale.exp().float() * image @ text.t()
        return {"logits": logits, "image_features": image, "text_features": text}

    def prompt_state_dict(self) -> dict[str, Tensor]:
        """Return only trainable prompts/projections, independent of vocabulary."""
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def load_prompt_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Validate all prompt entries before atomically copying their values."""
        if not isinstance(state, Mapping):
            raise TypeError("prompt state must be a mapping of exact trainable names")
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != set(parameters):
            missing = sorted(set(parameters) - set(state))
            extra = sorted(set(state) - set(parameters))
            raise ValueError(f"prompt state key mismatch; missing={missing}, extra={extra}")
        converted: dict[str, Tensor] = {}
        for name, parameter in parameters.items():
            value = state[name]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating tensor")
            if value.shape != parameter.shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(parameter.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            destination = value.to(device=parameter.device, dtype=parameter.dtype)
            if not torch.isfinite(destination).all():
                raise ValueError(f"{name} overflows the destination dtype")
            converted[name] = destination
        for name, parameter in parameters.items():
            parameter.copy_(converted[name])
