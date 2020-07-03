class FEL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FEL, self).__init__()
        self.conv = torch.nn.Conv1d(input_dim, output_dim, 1)
        self.bn = nn.BatchNorm1d(output_dim)
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x):
        n_pts = x.size()[2] # input shape is (B, C(channels), N(points))
        point_feature = x
        x = F.relu(self.bn(self.conv(x)))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, output_dim, 1).repeat(1, 1, n_pts)
        return torch.cat([point_feature, global_feature], 1)
    
class PSNET(nn.Module):
    def __init__(self, K=2):
        super(PSNET, self).__init__()
        self.FEL1 = FEL(3, 32)
        self.FEL2 = FEL(64, 128)
        self.FEL3 = FEL(256, 512)
        self.FEL4 = FEL(512, 1024)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, K)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.FEL1(x)
        x = self.FEL2(x)
        x = self.FEL3(x)
        x = self.FEL4(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

def gram_matrix(input_feature):
    batch = input_feature.size()[0]
    channels = input_feature.size()[1]
    n_pts = input_feature.size()[2]
    input_feature = input_feature.view(batch * channels, n_pts)
    G = torch.mm(input_feature, input_feature.t())
    G = G.div(batch * channels * n_pts)
    return G