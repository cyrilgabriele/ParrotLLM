"""Optimizer must not apply weight decay to embeddings, norms, or biases.

Reference: nanoGPT model.py::configure_optimizers; GPT-3 paper App. B;
LLaMA impl in HF transformers (LayerNorm + bias exempt via get_parameter_names).
"""
import torch

from src.model import ParrotLLM
from src.training.trainer import build_optimizer


def _build_tiny_model():
    cfg = {"model": {
        "vocab_size": 128, "d_model": 32, "n_layers": 2, "n_heads": 4,
        "d_ff": 64, "context_length": 16, "bias": False,
        "dropout": 0.0, "rope_theta": 10000.0,
        "gradient_checkpointing": False,
    }}
    return ParrotLLM(cfg)


def test_token_embedding_excluded_from_weight_decay():
    model = _build_tiny_model()
    tc = {"learning_rate": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95}
    opt = build_optimizer(model, tc)
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert id(model.tok_emb.weight) in no_decay_ids, (
        "token embedding should be in the no-decay group (canonical: nanoGPT)"
    )
    assert id(model.tok_emb.weight) not in decay_ids


def test_rmsnorm_weights_excluded_from_weight_decay():
    model = _build_tiny_model()
    tc = {"learning_rate": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95}
    opt = build_optimizer(model, tc)
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    norm_params = [
        p for n, p in model.named_parameters()
        if ("ln_" in n or "_norm" in n) and n.endswith("weight")
    ]
    assert len(norm_params) > 0, "test setup error: no norm params found"
    for p in norm_params:
        assert id(p) in no_decay_ids, "RMSNorm weights must not be decayed"


def test_qkv_and_ffn_weights_still_decayed():
    model = _build_tiny_model()
    tc = {"learning_rate": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95}
    opt = build_optimizer(model, tc)
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    matmul_params = [
        p for n, p in model.named_parameters()
        if any(s in n for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                "o_proj.weight", "gate_proj.weight",
                                "up_proj.weight", "down_proj.weight"))
    ]
    for p in matmul_params:
        assert id(p) in decay_ids, "matmul weights must remain in decay group"
