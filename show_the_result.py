import numpy as np
import Visualization
from plyfile import PlyData, PlyElement
from random import sample

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

content_point_cloud = np.asarray(content_point_cloud_temp)
style_point_cloud = np.asarray(style_point_cloud_temp)

'''
Visualization.show_point_cloud_three_view(content_point_cloud, 'content.jpg')
Visualization.show_point_cloud_three_view(style_point_cloud, 'style.jpg')
'''
for e in range(800, 4001, 100):
	print(e)
	epoch = str(e)
	point_cloud = np.load('./style transfer/style transfer' + epoch + '.npy')
	Visualization.show_point_cloud_three_view(point_cloud.T, './result/epoch' + epoch + '.jpg')