"""
ResNet-18 implementation for CIFAR-100
Adapted for 32x32 images (original ResNet is for 224x224 ImageNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-100 (32x32 images)
    
    Architecture:
    - Initial conv: 3x3, stride 1 (no max pooling for small images)
    - 4 layer groups: [64, 128, 256, 512] channels
    - Each group has 2 BasicBlocks
    - Global average pooling
    - Fully connected classifier
    """
    
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (adapted for 32x32, not 224x224)
        # No stride=2 or max pooling to preserve spatial resolution (see \utils\RF_search.py and report as for justification)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers (4 groups of 2 blocks each)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

        # Layer inspector: hooks to capture block features for OOD/NC analysis
        self._features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture block outputs"""
        def get_hook(name):
            def hook(module, input, output):
                self._features[name] = output.detach()
            return hook
        
        self.layer1.register_forward_hook(get_hook('layer1'))
        self.layer2.register_forward_hook(get_hook('layer2'))
        self.layer3.register_forward_hook(get_hook('layer3'))
        self.layer4.register_forward_hook(get_hook('layer4'))
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a layer group with multiple blocks"""
        layers = []
        
        # First block may downsample
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Kaiming initialization for conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: input images [B, 3, 32, 32]
            return_features: if True, return (logits, features)
        
        Returns:
            logits [B, num_classes] or (logits, features)
        """
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # [B, 512]
        
        # Classifier
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_features(self, x):
        """Extract penultimate layer features (before classifier)"""
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features
    
    def get_classifier_weights(self):
        """Get classifier weight matrix [num_classes, 512]"""
        return self.fc.weight.data
    
    def get_block_features(self, x, block_name='layer4'):
        """
        Extract features from a specific block
        
        Args:
            x: input images [B, 3, H, W]
            block_name: 'layer1', 'layer2', 'layer3', or 'layer4'
        
        Returns:
            features from specified block [B, C, H, W]
        """
        with torch.no_grad():
            _ = self.forward(x)
        return self._features.get(block_name)

    def get_all_features(self, x):
        """
        Get features from all layers in a single forward pass

        Args:
            x: input images [B, 3, H, W]

        Returns:
            dict with:
                'layer1': [B, 64, 32, 32]
                'layer2': [B, 128, 16, 16]
                'layer3': [B, 256, 8, 8]
                'layer4': [B, 512, 4, 4]
                'penultimate': [B, 512] (after avgpool)
                'logits': [B, 100]
        """
        with torch.no_grad():
            logits, penultimate = self.forward(x, return_features=True)

        return {
            'layer1': self._features['layer1'],
            'layer2': self._features['layer2'],
            'layer3': self._features['layer3'],
            'layer4': self._features['layer4'],
            'penultimate': penultimate,
            'logits': logits
        }


def test_resnet18():
    """Quick test of ResNet-18"""
    model = ResNet18(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    
    # Test forward pass
    logits = model(x)
    assert logits.shape == (4, 100), f"Expected (4, 100), got {logits.shape}"
    
    # Test feature extraction
    features = model.get_features(x)
    assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"
    
    # Test classifier weights
    weights = model.get_classifier_weights()
    assert weights.shape == (100, 512), f"Expected (100, 512), got {weights.shape}"
    
    # Test block features (hooks)
    block4_feats = model.get_block_features(x, 'layer4')
    assert block4_feats.shape == (4, 512, 4, 4), f"Expected (4, 512, 4, 4), got {block4_feats.shape}"
    
    # Test all features
    all_feats = model.get_all_features(x)
    assert all_feats['layer1'].shape == (4, 64, 32, 32), f"Layer1 shape mismatch"
    assert all_feats['layer2'].shape == (4, 128, 16, 16), f"Layer2 shape mismatch"
    assert all_feats['layer3'].shape == (4, 256, 8, 8), f"Layer3 shape mismatch"
    assert all_feats['layer4'].shape == (4, 512, 4, 4), f"Layer4 shape mismatch"
    assert all_feats['penultimate'].shape == (4, 512), f"Penultimate shape mismatch"
    assert all_feats['logits'].shape == (4, 100), f"Logits shape mismatch"
    
    print("Hooks working correctly!")

    print("ResNet18 tests passed!")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_resnet18()
