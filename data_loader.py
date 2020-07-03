import numpy as np
import json
import torch
import os
from torch.utils import data
from plyfile import PlyData, PlyElement

class densepointdataset(data.Dataset):
	def __init__(self, mode, root = './DensePoint/'):

		self.root = root
		self.mode = mode
		input_file = open(self.root + self.mode + '.json')
		json_array = json.load(input_file)
		self.categories = os.listdir(self.root)
		self.point_cloud_path = []
		self.point_cloud_label = []
		
		for i in json_array:
			#get categroy folder name
			for j in self.categories:
				if j.find(i) != -1:
					category = j
					break

			for e in json_array[i]:
				self.point_cloud_label.append(i)
				self.point_cloud_path.append(self.root + category + '/ply_file/' + e + '.ply')

		self.point_cloud_label = self.point_cloud_label[:10]
		self.point_cloud_path = self.point_cloud_path[:10]
		for e in self.point_cloud_path:
			a = PlyData.read(e)
			print(len(a.elements[0][:]))

	def __len__(self):
		return len(self.point_cloud_label)

	def __getitem__(self, index):
		
		point_cloud = PlyData.read(self.point_cloud_path[index])
		point_cloud_info = []
		
		#six dim XYZ RGB
		for e in point_cloud.elements[0]:
			point_cloud_info.append([e[i] for i in range(0, 6)])
		
		point_cloud_info = np.asarray(point_cloud_info).T

		label = self.point_cloud_label[index]
		return point_cloud_info, label


if __name__ == '__main__':

	test = densepointdataset('test')
	test.__getitem__(10)