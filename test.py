import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 38 * 38, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 38 * 38)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


model = STN()
loss = nn.SmoothL1Loss()
optim = torch.optim.Adam(model.params, lr=0.001)


class AlignDataset(Dataset):
    def __init__(self, root):
        self.file = 
        pass

    def __len__(self):
        pass
