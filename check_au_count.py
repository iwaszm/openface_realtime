import torch
import sys

try:
    model_path = "./weights/MTL_backbone.pth"
    print(f"Checking model: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Check for au_regressor keys to determine count
    # Keys look like: au_regressor.class_linears.0.weight, .1.weight, etc.
    max_idx = -1
    for k in state_dict.keys():
        if "au_regressor.class_linears" in k and ".weight" in k:
            # Extract index
            parts = k.split('.')
            # au_regressor.class_linears.0.weight -> index is 3rd part usually?
            # Let's find numeric part
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    if idx > max_idx:
                        max_idx = idx
    
    au_count = max_idx + 1
    print(f"✅ Found {au_count} AUs in the weight file.")
    
except Exception as e:
    print(f"❌ Error checking model: {e}")
