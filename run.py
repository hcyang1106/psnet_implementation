import torch
import torch.nn as nn
import data_loader
import torch.utils.data as Data
from plyfile import PlyData, PlyElement
import numpy as np
from random import sample
import copy
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

    def forward(self, x_color, x_geo):
        color_list = []
        geo_list = []
        
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
        
        return color_list, geo_list

model = torch.load('./save_model/PSNet_10_1', map_location='cpu')
torch.save(model.state_dict(), "PSNet_state_dict") 

model = PSNET()
model.load_state_dict(torch.load("PSNet_state_dict"))
model.cuda()
model.eval()

def gram_matrix(input_feature):
    batch, channels, n_pts = input_feature.size()
    input_feature = input_feature.view(batch * channels, n_pts)
    gram = torch.mm(input_feature.t(), input_feature)
    return gram.div(batch * channels * n_pts)

#for param in model.parameters():   # fix all the weights
#    param.requires_grad = False

content_point_cloud_data = PlyData.read('1c111a837580fda6c3bd24f986301745.ply')
style_point_cloud_data = PlyData.read('dee83e7ab66cd504d88da0963249578d.ply')

content_point_cloud_temp = []
for e in content_point_cloud_data.elements[0]:
    content_point_cloud_temp.append([e[i] for i in range(0, 6)])
    
style_point_cloud_temp = []
for e in style_point_cloud_data.elements[0]:
    style_point_cloud_temp.append([e[i] for i in range(0, 6)])

content_point_cloud_temp = sample(content_point_cloud_temp, 5000)
style_point_cloud_temp = sample(style_point_cloud_temp, 5000)
content_point_cloud = torch.Tensor(np.asarray(content_point_cloud_temp).T).cuda().unsqueeze(0)
style_point_cloud = torch.Tensor(np.asarray(style_point_cloud_temp).T).cuda().unsqueeze(0)

content_color_feature, content_geo_feature = model(content_point_cloud[:, :3, :], content_point_cloud[:, 3:, :]) 
style_color_feature, style_geo_feature = model(style_point_cloud[:, :3, :], style_point_cloud[:, 3:, :]) 
#print(content_point_cloud.shape)

# current_point_cloud initialization
criterion = torch.nn.MSELoss()
current_geo_part = torch.nn.Parameter(content_point_cloud.data[:, 3:, :], requires_grad=True)
current_color_part = torch.nn.Parameter(content_point_cloud.data[:, :3, :], requires_grad=True)
current_geo_part = current_geo_part.cuda()
current_color_part = current_color_part.cuda()
#geo_optimizer = torch.optim.SGD([current_geo_part.requires_grad_()], lr=1e-2)
geo_optimizer = torch.optim.Adam([current_geo_part.requires_grad_()], lr=1e-1, betas=(0.9, 0.999))
color_optimizer = torch.optim.Adam([current_color_part.requires_grad_()], lr=1e-1, betas=(0.9, 0.999))
#color_optimizer = torch.optim.SGD([current_color_part.requires_grad_()], lr=1e-2)
for epoch in range(1, 4000 + 1):
    '''
    current_geo_part = torch.nn.Parameter(current_point_cloud.data[:, 3:, :])
    current_color_part = torch.nn.Parameter(current_point_cloud.data[:, :3, :])
    geo_optimizer = torch.optim.SGD([current_geo_part], lr=1e-2)
    color_optimizer = torch.optim.SGD([current_color_part], lr=1e-2)
    '''
    geo_optimizer.zero_grad() 
    color_optimizer.zero_grad() 

    current_color_feature, current_geo_feature = model(current_color_part, current_geo_part) 

    style_geo_weight = 1.0
    style_color_weight = 100.0
    # geo part

    # content
    geo_content_loss = 0.0
    for layer_id in range(1):#len(current_geo_feature)):
        geo_content_loss += criterion(current_geo_feature[layer_id], content_geo_feature[layer_id])

    # style
    geo_style_loss = 0.0
    for layer_id in range(1):#len(current_geo_feature)):
        geo_style_loss += criterion(gram_matrix(current_geo_feature[layer_id]), gram_matrix(style_geo_feature[layer_id]))
    
    geo_total_loss = geo_content_loss + style_geo_weight * geo_style_loss
    print(geo_total_loss)
    geo_total_loss.backward(retain_graph=True)
    geo_optimizer.step()

    # color part

    # content
    color_content_loss = 0.0
    for layer_id in range(1):#len(current_color_feature)):
        color_content_loss += criterion(current_color_feature[layer_id], content_color_feature[layer_id])

    # style
    color_style_loss = 0.0
    for layer_id in range(1):#len(current_color_feature)):
        color_style_loss += criterion(gram_matrix(current_color_feature[layer_id]), gram_matrix(style_color_feature[layer_id]))

    color_total_loss = color_content_loss + style_color_weight * color_style_loss
    print(color_total_loss)
    color_total_loss.backward(retain_graph=True)

    color_optimizer.step()

    current_point_cloud = torch.cat((current_color_part.data, current_geo_part.data), 1)
    print(epoch, 'epoch finished.')
    
    if epoch % 100 == 0 :
        np.save("./style transfer/" + "style transfer" + str(epoch), content_point_cloud.squeeze(0).cpu().detach().numpy())
    