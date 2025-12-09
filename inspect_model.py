import torch
import torch.nn as nn
import torchvision.models as models

# Load the saved model
state = torch.load('d:/IIT/Core/models/best_model.pth', map_location='cpu', weights_only=False)

print("="*80)
print("MODEL INSPECTION")
print("="*80)

# Check what's in the state dict
print("\n1. State dict keys:")
for key in state.keys():
    print(f"   - {key}")

# Check model architecture info
if 'epoch' in state:
    print(f"\n2. Training info:")
    print(f"   - Epoch: {state['epoch']}")
if 'val_loss' in state:
    print(f"   - Val Loss: {state['val_loss']}")
if 'val_accuracy' in state:
    print(f"   - Val Accuracy: {state['val_accuracy']}")
if 'backbone' in state:
    print(f"   - Backbone: {state['backbone']}")
if 'num_classes' in state:
    print(f"   - Num Classes: {state['num_classes']}")

# Try to load with resnext101_32x8d
print("\n3. Testing model loading with resnext101_32x8d:")
try:
    class MultiLabelResNet(nn.Module):
        def __init__(self, num_classes, backbone='resnext101_32x8d', pretrained=False):
            super(MultiLabelResNet, self).__init__()
            if backbone == 'resnext101_32x8d':
                self.base_model = models.resnext101_32x8d(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Freeze all parameters initially
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            # Replace the final classification layer
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes)
            )
        
        def forward(self, x):
            return self.base_model(x)
    
    model = MultiLabelResNet(num_classes=14, backbone='resnext101_32x8d', pretrained=False)
    
    # Try strict loading
    print("\n   Trying strict=True loading...")
    try:
        model.load_state_dict(state['model_state_dict'], strict=True)
        print("   ✓ Strict loading SUCCESSFUL!")
    except Exception as e:
        print(f"   ✗ Strict loading FAILED: {str(e)[:200]}")
        
        # Try strict=False
        print("\n   Trying strict=False loading...")
        result = model.load_state_dict(state['model_state_dict'], strict=False)
        print(f"   ✓ Loaded with strict=False")
        if result.missing_keys:
            print(f"   Missing keys: {result.missing_keys[:10]}")
        if result.unexpected_keys:
            print(f"   Unexpected keys: {result.unexpected_keys[:10]}")
    
    # Test inference with a dummy input
    print("\n4. Testing inference with dummy input:")
    model.eval()
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print(f"   Output mean: {output.mean().item():.2f}")
    print(f"   Output std: {output.std().item():.2f}")
    
    # Check if FC layer weights look reasonable
    fc_weight = model.base_model.fc[1].weight
    fc_bias = model.base_model.fc[1].bias
    print(f"\n5. FC layer statistics:")
    print(f"   Weight range: [{fc_weight.min().item():.4f}, {fc_weight.max().item():.4f}]")
    print(f"   Weight mean: {fc_weight.mean().item():.4f}")
    print(f"   Bias range: [{fc_bias.min().item():.4f}, {fc_bias.max().item():.4f}]")
    print(f"   Bias mean: {fc_bias.mean().item():.4f}")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
