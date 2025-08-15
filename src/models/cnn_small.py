import torch.nn as nn

def create_fruit_cnn(pooling="max", in_channels=1, num_classes=8):
    if pooling not in {"max", "adaptiveavg"}:
        raise ValueError("pooling must be 'max' or 'adaptiveavg'")

    pool1 = nn.MaxPool2d(2, 2)
    pool2 = nn.MaxPool2d(2, 2)
    if pooling == "adaptiveavg":
        # use deterministic average pooling instead of AdaptiveAvgPool2d
        pool3 = nn.AvgPool2d(2, 2)
    else:
        pool3 = nn.MaxPool2d(2, 2)

    class FruitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), pool1,
                nn.Conv2d(16, 32, 3, padding=1),          nn.ReLU(), pool2,
                nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(), pool3,
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*12*12, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.fc(self.conv(x))
    return FruitCNN()
