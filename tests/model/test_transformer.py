import torch
import pytest
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
    # tok_emb: 128 * 64 = 8192
    # pos_emb: 64 * 64 = 4096
    # 2 blocks * (
    #   ln_1 (64 weight, 64 bias) = 128
    #   attn (4 heads * (64*64)) = 16384
    #   ln_2 (64 weight, 64 bias) = 128
    #   mlp (2 * (64*256)) = 32768
    # ) = 2 * (128 + 16384 + 128 + 32768) = 2 * 49408 = 98816
    # ln_f (64 weight, 64 bias) = 128
    # Total: 8192 + 4096 + 98816 + 128 = 111232
    assert count == 111232

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
