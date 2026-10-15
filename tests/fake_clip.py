from __future__ import annotations
import hashlib
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 2, dropout=0.)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
    def forward(self, x):
        query = self.ln(x)
        x = x + self.attn(query, query, query, need_weights=False)[0]
        return x + self.mlp(x)

class Transformer(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.resblocks = nn.ModuleList([Block(width) for _ in range(depth)])
    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x

class Visual(nn.Module):
    def __init__(self, width, embedding, depth, input_size=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=input_size//2, stride=input_size//2, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width) * .1)
        self.positional_embedding = nn.Parameter(torch.randn(5, width) * .02)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, depth)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, embedding) * .1)
    def forward(self, images):
        x = self.conv1(images).flatten(2).transpose(1, 2)
        cls = self.class_embedding.reshape(1, 1, -1).expand(len(images), 1, -1)
        x = self.ln_pre(torch.cat([cls, x], dim=1) + self.positional_embedding)
        x = self.transformer(x.transpose(0, 1))[0]
        return self.ln_post(x) @ self.proj

class FakeClip(nn.Module):
    """Small, genuinely differentiable CLIP-shaped CPU object, not real CLIP."""
    def __init__(self, depth=12, text_width=8, visual_width=12, embedding=8, input_size=8):
        super().__init__()
        self.visual = Visual(visual_width, embedding, depth, input_size)
        self.transformer = Transformer(text_width, depth)
        self.token_embedding = nn.Embedding(100, text_width)
        self.positional_embedding = nn.Parameter(torch.randn(32, text_width) * .02)
        self.ln_final = nn.LayerNorm(text_width)
        self.text_projection = nn.Parameter(torch.randn(text_width, embedding) * .1)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
    def encode_image(self, images):
        return self.visual(images)
    def encode_text(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding
        x = self.ln_final(self.transformer(x.transpose(0, 1)).transpose(0, 1))
        return x[torch.arange(len(tokens)), tokens.argmax(-1)] @ self.text_projection

def tokenize(texts, context_length=32, truncate=False):
    tokens = torch.zeros(len(texts), context_length, dtype=torch.long)
    for index, text in enumerate(texts):
        words = text.split()
        if len(words) > context_length - 2:
            raise ValueError("fake tokenizer context exceeded")
        tokens[index, 0] = 1
        for pos, word in enumerate(words, 1):
            tokens[index, pos] = 2 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 97
        tokens[index, len(words) + 1] = 99
    return tokens

CLIP_MODULE = SimpleNamespace(tokenize=tokenize)

def config(method, **settings):
    return {"model": {"prompt_length": 16, "depth": 12, "shallow_layers": 6,
                      "alignment_metric": "cosine", "alignment_weight": 1.,
                      "text_alignment_weight": 1., "image_alignment_weight": 1.},
            "method": {"name": method, "enabled": True, **settings}}

def vocabulary(model, count=3):
    model.set_classnames(["class%d" % i for i in range(count)],
                         ["a photo of land class%d." % i for i in range(count)])

def make_model(model_class, settings=None, seed=123):
    torch.manual_seed(seed)
    clip = FakeClip()
    model = model_class(clip, CLIP_MODULE, settings or config(model_class.method_name))
    vocabulary(model)
    return model
