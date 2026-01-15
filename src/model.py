import torch
import torch.nn as nn
import torchvision.models as models

class PneumoNet(nn.Module):
    def __init__(self):
        super(PneumoNet, self).__init__()
        # 1. Take a pre-trained 'brain' (ResNet18) that already knows how to see edges
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # 2. ResNet was built to see 1000 things (dogs, cars, etc.)
        # We change the final layer to see only ONE thing: Pneumonia (0 to 1)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)
        
        # 3. Sigmoid turns the result into a percentage (e.g., 0.85 = 85% chance)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # This is the "Forward Pass" - how the image travels through the brain
        x = self.backbone(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    # TEST: Let's see if the brain works
    model = PneumoNet()
    
    # Create a "Fake" X-ray (just random numbers) to see if it crashes
    # Shape: [1 image, 3 colors, 224 pixels wide, 224 pixels high]
    fake_xray = torch.randn(1, 3, 224, 224) 
    
    prediction = model(fake_xray)
    print(f"✅ AI Brain Created!")
    print(f"Prediction for fake image: {prediction.item():.4f} (Probability)")