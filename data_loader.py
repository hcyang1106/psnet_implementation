import numpy as np
import json
import torch
import os
from torch.utils import data
from plyfile import PlyData, PlyElement
from random import sample

labelset = {'04225987' : 0, '03790512' : 1, '02954340' : 2, '03261776' : 3, '02958343' : 4, '03001627' : 5,\
 '03642806' : 6, '04099429' : 7, '03636649' : 8, '04379243' : 9, '03948459' : 10, '03624134' : 11,\
 '03467517' : 12, '02691156' : 13, '03797390' : 14, '02773838' : 15}

labelcount = {'04225987' : [], '03790512' : [], '02954340' : [], '03261776' : [], '02958343' : [], '03001627' : [],\
 '03642806' : [], '04099429' : [], '03636649' : [], '04379243' : [], '03948459' : [], '03624134' : [],\
 '03467517' : [], '02691156' : [], '03797390' : [], '02773838' : []}
class densepointdataset(data.Dataset):
	def __init__(self, mode, root = './DensePoint/'):

		self.root = root
		self.mode = mode
		input_file = open(self.root + self.mode + '.json')
		json_array = json.load(input_file)
		self.categories = os.listdir(self.root)
		self.point_cloud_path = []
		self.point_cloud_label = []
		self.labelset = dict()
		for i in json_array:
			#get categroy folder name
			for j in self.categories:
				if j.find(i) != -1:
					category = j
					break

			for e in json_array[i]:
				labelcount[i].append(self.root + category + '/ply_file/' + e + '.ply')
				self.point_cloud_label.append(labelset[i])
				self.point_cloud_path.append(self.root + category + '/ply_file/' + e + '.ply')

		self.sample_data = []
		self.sample_label = []
		for e in labelcount.keys():
			if len(labelcount[e]) > 320:
				self.sample_data += sample(labelcount[e], 320)
				self.sample_label += [labelset[e]] * 320
			else:
				self.sample_data += labelcount[e]
				self.sample_label += [labelset[e]] * len(labelcount[e])
		'''
		self.point_cloud_label = self.point_cloud_label[:3]
		self.point_cloud_path = self.point_cloud_path[:3]
		self.sample_data = self.sample_data[:3]
		self.sample_label = self.sample_label[:3]
		'''
		self.min_point = 39690
		self.min_point = 5000
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

		#return len(self.sample_label)
		#return len(self.point_cloud_label)
		if self.mode == 'train':
			#return len(self.sample_label)
			return len(self.point_cloud_label)
		elif self.mode == 'test':
			return len(self.point_cloud_label)

	def __getitem__(self, index):
		
		if self.mode == 'train':
			#point_cloud = PlyData.read(self.sample_data[index])
			point_cloud = PlyData.read(self.point_cloud_path[index])
		elif self.mode == 'test':
			point_cloud = PlyData.read(self.point_cloud_path[index])

		point_cloud_temp = []
		#six dim XYZ RGB
		#normalize to 1 ~ -1
		for e in point_cloud.elements[0]:
			point_cloud_temp.append([e[i] for i in range(0, 6)])

		point_cloud_info = []
		point_cloud_sample = sample(point_cloud_temp, self.min_point)
		label = int(self.point_cloud_label[index])
		'''
		min_max = np.asarray([[2e9, 2e9, 2e9], [0, 0 , 0]])
		for e in point_cloud_sample :
			for i in range(0, 3):
				if e[i] > min_max[1][i]:
					min_max[1][i] = e[i]
				if e[i] < min_max[0][i]:
					min_max[0][i] = e[i]

		for e in point_cloud_sample:	
			temp = [(e[i] - min_max[0][i]) / (min_max[1][i] - min_max[0][i]) * 2 - 1 for i in range(0, 3)]
			temp+=[(e[i]/255.0 - 0.5) *2 for i in range(3, 6)]
			point_cloud_info.append(temp)
		'''
		point_cloud_info = point_cloud_sample
		point_cloud_info = np.asarray(point_cloud_info).T

		return point_cloud_info, label


if __name__ == '__main__':

	test = densepointdataset('train')
	test.__getitem__(10)