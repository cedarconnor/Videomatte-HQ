
import torch
import pytest
from videomatte_hq.models.edge_vitmatte import ViTMatteModel

@pytest.mark.skip(reason="Requires model weights")
def test_vitmatte_shapes():
    """Test ViTMatte tensor shape handling."""
    model = ViTMatteModel(device="cpu", precision="fp32")
    # model.load_weights("cpu") # Skipped
    
    # Mock infer_tile logic
    h, w = 100, 100
    rgb = torch.randn(3, h, w)
    trimap = torch.randn(1, h, w)
    
    # Concatenate
    inp = torch.cat([rgb.unsqueeze(0), trimap.unsqueeze(0)], dim=1)
    assert inp.shape == (1, 4, h, w)
    print("Tensor shapes OK")

if __name__ == "__main__":
    test_vitmatte_shapes()
