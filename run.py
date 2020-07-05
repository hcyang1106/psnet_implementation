import PSNet
import torch
import data_loader

model = torch.load('PSNet_10_1')

def gram_matrix(input_feature):
    batch, channels, n_pts = input_feature.size()
    input_feature = input_feature.view(batch * channels, n_pts)
    gram = torch.mm(input_feature.t(), input_feature)
    return gram.div(batch * channels * n_pts)

for param in model.parameters():   # fix all the weights
    param.requires_grad = False

train_loader = Data.DataLoader(data_loader.densepointdataset(mode = 'train'), batch_size = 1, shuffle = True)
test_loader = Data.DataLoader(data_loader.densepointdataset(mode = 'test'), batch_size = 1, shuffle = True)


# current_point_cloud initialization
current_point_cloud = content_point_cloud

for epoch in range(100):
    current_geo_part = torch.nn.Parameter(current_point_cloud[:, 3:, :].data)
    current_color_part = torch.nn.Parameter(current_point_cloud[:, :3, :].data)
    geo_optimizer = torch.optim.SGD([current_geo_part])
    color_optimizer = torch.optim.SGD([current_color_part])


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
        geo_content_loss += torch.nn.MSELoss(current_geo_feature[layer_id], content_geo_feature[layer_id])

    # style
    geo_style_loss = 0.0
    for layer_id in range(len(current_geo_feature)):
        geo_style_loss += torch.nn.MSELoss(gram_matrix(current_geo_feature[layer_id]), gram_matrix(content_geo_feature[layer_id]))

    geo_total_loss = content_weight * geo_content_loss + style_weight * geo_style_loss
    geo_total_loss.backward()

    geo_optimizer.step()

    # color part

    # content
    color_content_loss = 0.0
    for layer_id in range(len(current_color_feature)):
        color_content_loss += torch.nn.MSELoss(current_color_feature[layer_id], content_color_feature[layer_id])

    # style
    color_style_loss = 0.0
    for layer_id in range(len(current_color_feature)):
        color_style_loss += torch.nn.MSELoss(gram_matrix(current_color_feature[layer_id]), gram_matrix(content_color_feature[layer_id]))

    color_total_loss = content_weight * color_content_loss + style_weight * color_style_loss
    color_total_loss.backward()

    color_optimizer.step()

    current_point_cloud = torch.concat((current_color_part.data, current_geo_part.data), 1)


    

