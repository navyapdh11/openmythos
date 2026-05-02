import torch
from hypothesis import given, strategies as st, settings
from openmythos import OpenMythos

@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=64),
    loop_iters=st.integers(min_value=0, max_value=4)
)
def test_openmythos_output_shape(batch_size: int, seq_len: int, loop_iters: int) -> None:
    dim = 64
    vocab_size = 1000
    model = OpenMythos(
        dim=dim,
        num_heads=4,
        latent_dim=16,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        vocab_size=vocab_size
    )
    
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(tokens, loop_iters=loop_iters)
    
    assert output.shape == (batch_size, seq_len, vocab_size)

def test_openmythos_loop_consistency() -> None:
    """Test that zero loops still works (only prelude + coda)."""
    model = OpenMythos(dim=64, num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1)
    tokens = torch.randint(0, 50257, (1, 8))
    
    out_0 = model(tokens, loop_iters=0)
    out_1 = model(tokens, loop_iters=1)
    
    assert out_0.shape == out_1.shape
    # Outputs should be different because of the extra loop
    assert not torch.allclose(out_0, out_1)
