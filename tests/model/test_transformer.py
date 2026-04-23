import torch
import pytest
import torch.nn.functional as F
from src.model.transformer import ParrotLLM

@pytest.fixture
def small_config():
    return {
        "model": {
            "vocab_size": 128,
            "context_length": 64,
            "d_model": 64,
            "n_heads": 4,
            "d_ff": 256,
            "n_layers": 2,
            "dropout": 0.0,
            "bias": False
        }
    }

def test_model_forward_shape(small_config):
    model = ParrotLLM(small_config)
    B, T = 4, 16
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    
    logits, loss = model(idx)
    
    assert logits.shape == (B, T, small_config["model"]["vocab_size"])
    assert loss is None

def test_model_forward_with_targets(small_config):
    model = ParrotLLM(small_config)
    B, T = 4, 16
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    targets = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    
    logits, loss = model(idx, targets)
    
    assert logits.shape == (B, T, small_config["model"]["vocab_size"])
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # scalar

def test_chunked_loss_matches_full_loss(small_config):
    model = ParrotLLM(small_config)
    B, T = 4, 16
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    targets = torch.randint(0, small_config["model"]["vocab_size"], (B, T))

    logits, full_loss = model(idx, targets, z_loss_coeff=1e-4)
    chunked_logits, chunked_loss = model(
        idx,
        targets,
        return_logits=False,
        z_loss_coeff=1e-4,
        loss_chunk_rows=7,
    )

    assert logits.shape == (B, T, small_config["model"]["vocab_size"])
    assert chunked_logits is None
    assert torch.allclose(chunked_loss, full_loss, atol=1e-6, rtol=1e-5)


def test_chunked_masked_loss_matches_full_masked_loss(small_config):
    model = ParrotLLM(small_config)
    B, T = 4, 16
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    targets = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    loss_mask = torch.randint(0, 2, (B, T), dtype=torch.int64).to(torch.float32)
    loss_mask[0, 0] = 1.0

    logits, full_loss = model(
        idx,
        targets,
        loss_mask=loss_mask,
        z_loss_coeff=1e-4,
    )
    chunked_logits, chunked_loss = model(
        idx,
        targets,
        loss_mask=loss_mask,
        return_logits=False,
        z_loss_coeff=1e-4,
        loss_chunk_rows=7,
    )

    assert logits.shape == (B, T, small_config["model"]["vocab_size"])
    assert chunked_logits is None
    assert torch.allclose(chunked_loss, full_loss, atol=1e-6, rtol=1e-5)


def test_chunked_loss_recovers_from_simulated_mps_oom(small_config, monkeypatch):
    model = ParrotLLM(small_config)
    B, T = 4, 16
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    targets = torch.randint(0, small_config["model"]["vocab_size"], (B, T))

    _, baseline_loss = model(
        idx,
        targets,
        return_logits=False,
        loss_chunk_rows=2,
    )

    original_linear = F.linear

    def flaky_linear(input, weight, bias=None):
        if (
            input.dim() == 2
            and weight.data_ptr() == model.lm_head.weight.data_ptr()
            and input.size(0) > 1
        ):
            raise RuntimeError("MPS backend out of memory")
        return original_linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", flaky_linear)
    _, recovered_loss = model(
        idx,
        targets,
        return_logits=False,
        loss_chunk_rows=8,
    )

    assert torch.allclose(recovered_loss, baseline_loss, atol=1e-6, rtol=1e-5)

def test_weight_tying(small_config):
    model = ParrotLLM(small_config)
    assert model.tok_emb.weight is model.lm_head.weight

def test_causality(small_config):
    model = ParrotLLM(small_config)
    model.eval() # Disable dropout
    
    B, T = 1, 10
    idx = torch.randint(0, small_config["model"]["vocab_size"], (B, T))
    
    # Forward pass on full sequence
    logits1, _ = model(idx)
    
    # Change the last token and check if previous logits change
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % small_config["model"]["vocab_size"]
    
    logits2, _ = model(idx2)
    
    # All but the last position's logits should be identical
    assert torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-6)
    # The last position should be different
    assert not torch.allclose(logits1[:, -1, :], logits2[:, -1, :], atol=1e-6)

def test_parameter_count(small_config):
    model = ParrotLLM(small_config)
    count = model.count_parameters()
    # tok_emb: 128 * 64 = 8192 (lm_head is weight-tied, not counted)
    # 2 blocks * (
    #   ln_1: RMSNorm(64) = 64
    #   attn: q/k/v/o_proj (4 * 64*64) = 16384 + q_norm(16) + k_norm(16) = 16416
    #   ln_1_out: RMSNorm(64) = 64  (Peri-LN post-norm)
    #   ln_2: RMSNorm(64) = 64
    #   mlp: SwiGLU gate/up/down (3 * 64*256) = 49152
    #   ln_2_out: RMSNorm(64) = 64  (Peri-LN post-norm)
    # ) = 2 * (64 + 16416 + 64 + 64 + 49152 + 64) = 2 * 65824 = 131648
    # ln_f: RMSNorm(64) = 64
    # Total: 8192 + 131648 + 64 = 139904
    assert count == 139904

def test_overfit_single_batch(small_config):
    model = ParrotLLM(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2) # Higher lr for fast overfit
    
    B, T = 1, 4
    # Simple repeating sequence
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 1]], dtype=torch.long)
    
    initial_loss = None
    for i in range(100):
        logits, loss = model(idx, targets)
        if initial_loss is None:
            initial_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
    assert final_loss < initial_loss
    assert final_loss < 0.01 # Should be very low for this trivial case
