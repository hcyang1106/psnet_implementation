import numpy as np
import json
import torch
import os
from torch.utils import data
from plyfile import PlyData, PlyElement
from random import sample

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

		self.point_cloud_label = self.point_cloud_label[:3]
		self.point_cloud_path = self.point_cloud_path[:3]
		self.min_point = 39690
		self.max_point = 40370
		'''
		for e in self.point_cloud_path:
			a = PlyData.read(e)
			temp = len(a.elements[0][:])
			if temp < min_point:
				min_point = temp
			if temp > max_point:
				max_point = temp
		print(min_point, max_point)
		'''

	def __len__(self):
		return len(self.point_cloud_label)

	def __getitem__(self, index):
		
		point_cloud = PlyData.read(self.point_cloud_path[index])
		point_cloud_info = []
		#six dim XYZ RGB
		#normalize to 1 ~ -1
		min_max = np.asarray([[2e9, 2e9, 2e9], [0, 0 , 0]])
		for e in point_cloud.elements[0]:
			for i in range(0, 3):
				if e[i] > min_max[1][i]:
					min_max[1][i] = e[i]
				if e[i] < min_max[0][i]:
					min_max[0][i] = e[i]

		for e in point_cloud.elements[0]:	
			temp = [(e[i] - min_max[0][i]) / (min_max[1][i] - min_max[0][i]) * 2 - 1 for i in range(0, 3)]
			temp+=[(e[i]/255.0 - 0.5) *2 for i in range(3, 6)]
			point_cloud_info.append(temp)
		
		point_cloud_info = sample(point_cloud_info, self.min_point)
		point_cloud_info = np.asarray(point_cloud_info).T
		label = int(self.point_cloud_label[index])

		return point_cloud_info, label


if __name__ == '__main__':

	test = densepointdataset('test')
	test.__getitem__(10)