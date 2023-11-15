
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageFeature(nn.Module):
    def __init__(self, defaultFeatureSize=1024, device="cpu"):
        super().__init__()
        self.defaultFeatureSize = defaultFeatureSize
        self.linear = torch.nn.Linear(2048, self.defaultFeatureSize)
        self.relu = torch.nn.ReLU()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, X):
        X = X.to(device)
        output = self.relu(self.linear(X))
        return output, torch.mean(output, dim=1)  # batch * 196 *1024,  batch  * 1024

# if __name__ == "__main__":
