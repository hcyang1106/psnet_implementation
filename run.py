import PSNet
import torch
import torch.nn as nn
import data_loader
import torch.utils.data as Data
from plyfile import PlyData, PlyElement
import numpy as np

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
        color_list = []
        geo_list = []
        x_color = x[:, :3, :]
        x_geo = x[:, 3:, :]
        
        x_color = self.colorFEL1(x_color)
        color_list.append(x_color)
        x_color = self.colorFEL2(x_color)
        color_list.append(x_color)
        x_color = self.colorFEL3(x_color)
        color_list.append(x_color)
        x_color = self.colorFEL4(x_color)
        color_list.append(x_color)
        
        x_geo = self.geoFEL1(x_geo)
        geo_list.append(x_geo)
        x_geo = self.geoFEL2(x_geo)
        geo_list.append(x_geo)
        x_geo = self.geoFEL3(x_geo)
        geo_list.append(x_geo)
        x_geo = self.geoFEL4(x_geo)
        geo_list.append(x_geo)
        
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
        return x, color_list, geo_list

model = torch.load('PSNet_10_1')
torch.save(model.state_dict(), "PSNet_state_dict") 

model = PSNET().cuda()
model.load_state_dict(torch.load("PSNet_state_dict"))
model.eval()

def gram_matrix(input_feature):
    batch, channels, n_pts = input_feature.size()
    input_feature = input_feature.view(batch * channels, n_pts)
    gram = torch.mm(input_feature.t(), input_feature)
    return gram.div(batch * channels * n_pts)

#for param in model.parameters():   # fix all the weights
#    param.requires_grad = False

content_point_cloud_data = PlyData.read('1a1a5facf2b73e534e23e9314af9ae57.ply')
style_point_cloud_data = PlyData.read('1b5fc54e45c8768490ad276cd2af3a4.ply')

content_point_cloud_temp = []
for e in content_point_cloud_data.elements[0]:
    content_point_cloud_temp.append([e[i] for i in range(0, 6)])
    
style_point_cloud_temp = []
for e in style_point_cloud_data.elements[0]:
    style_point_cloud_temp.append([e[i] for i in range(0, 6)])

content_point_cloud = torch.Tensor(np.asarray(content_point_cloud_temp).T[:,:5000]).cuda()
style_point_cloud = torch.Tensor(np.asarray(style_point_cloud_temp).T[:,:5000]).cuda()

content_point_cloud = content_point_cloud.unsqueeze(0)
style_point_cloud = style_point_cloud.unsqueeze(0)

#print(content_point_cloud.shape)

# current_point_cloud initialization
current_point_cloud = content_point_cloud
criterion = torch.nn.MSELoss()

for epoch in range(100):
    current_geo_part = torch.nn.Parameter(current_point_cloud.data[:, 3:, :])
    current_color_part = torch.nn.Parameter(current_point_cloud.data[:, :3, :])
    geo_optimizer = torch.optim.SGD([current_geo_part], lr=1e-2)
    color_optimizer = torch.optim.SGD([current_color_part], lr=1e-2)


    geo_optimizer.zero_grad() 
    color_optimizer.zero_grad() 

    _, content_color_feature, content_geo_feature = model(content_point_cloud) 
    _, style_color_feature, style_geo_feature = model(style_point_cloud) 

    _, current_color_feature, current_geo_feature = model(current_point_cloud) 

    content_weight = 1.0
    style_weight = 1.0

    # geo part

    # content
    geo_content_loss = 0.0
    for layer_id in range(len(current_geo_feature)):
        geo_content_loss += criterion(current_geo_feature[layer_id], content_geo_feature[layer_id])

    # style
    geo_style_loss = 0.0
    for layer_id in range(len(current_geo_feature)):
        geo_style_loss += criterion(gram_matrix(current_geo_feature[layer_id]), gram_matrix(content_geo_feature[layer_id]))

    geo_total_loss = content_weight * geo_content_loss + style_weight * geo_style_loss
    geo_total_loss.backward()

    geo_optimizer.step()

    # color part

    # content
    color_content_loss = 0.0
    for layer_id in range(len(current_color_feature)):
        color_content_loss += criterion(current_color_feature[layer_id], content_color_feature[layer_id])

    # style
    color_style_loss = 0.0
    for layer_id in range(len(current_color_feature)):
        color_style_loss += criterion(gram_matrix(current_color_feature[layer_id]), gram_matrix(content_color_feature[layer_id]))

    color_total_loss = content_weight * color_content_loss + style_weight * color_style_loss
    color_total_loss.backward()

    color_optimizer.step()

    current_point_cloud = torch.cat((current_color_part.data, current_geo_part.data), 1)
    print(epoch, 'epoch finished.')
