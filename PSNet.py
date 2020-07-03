import torch.nn as nn
import torch

class FEL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FEL, self).__init__()
        self.conv = torch.nn.Conv1d(input_dim, output_dim, 1)
        self.bn = nn.BatchNorm1d(output_dim)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, x):

        n_pts = x.size()[2] # input shape is (B, C(channels), N(points))
        point_feature = x
        #x = F.relu(self.bn(self.conv(x)))
        x = self.activation(self.bn(self.conv(x)))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, self.output_dim, 1).repeat(1, 1, n_pts)
        return torch.cat([x, global_feature], 1)
    
class PSNET(nn.Module):
    def __init__(self, K = 16):
        super(PSNET, self).__init__()
        self.colorFEL1 = FEL(3, 32)
        self.colorFEL2 = FEL(64, 128)
        self.colorFEL3 = FEL(256, 512)
        self.colorFEL4 = FEL(1024, 1024)
        self.geoFEL1 = FEL(3, 32)
        self.geoFEL2 = FEL(64, 128)
        self.geoFEL3 = FEL(256, 512)
        self.geoFEL4 = FEL(1024, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(0.7)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.7)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.7)
            )
        self.fc4 = nn.Linear(64, K)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x_color = x[:, :3, :]
        x_geo = x[:, 3:, :]
        x_color = self.colorFEL1(x_color)
        x_color = self.colorFEL2(x_color)
        x_color = self.colorFEL3(x_color)
        x_color = self.colorFEL4(x_color)

        x_geo = self.geoFEL1(x_geo)
        x_geo = self.geoFEL2(x_geo)
        x_geo = self.geoFEL3(x_geo)
        x_geo = self.geoFEL4(x_geo)
        x = torch.cat([x_geo, x_color], dim = 1)
        x = torch.mean(x, dim = 2)
        '''
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        '''
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x #F.softmax(x, dim=1)

def gram_matrix(input_feature):
    batch = input_feature.size()[0]
    channels = input_feature.size()[1]
    n_pts = input_feature.size()[2]
    input_feature = input_feature.view(batch * channels, n_pts)
    G = torch.mm(input_feature, input_feature.t())
    G = G.div(batch * channels * n_pts)
    return G