import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import PSNet
import Visualization
import data_loader

if __name__ == '__main__':

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.load('./save_model/' + 'PSNet_10_1').to(device)
	cm_normalize = 'true'

	Batch_size = 16
	lr = 0.001
	epochs = 50

	optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999))
	test_loader = Data.DataLoader(data_loader.densepointdataset(mode = 'test'), \
	 batch_size = Batch_size, shuffle = True)#, collate_fn = my_collate)

	model.eval()
	print('test')
	test_label = []
	test_predict = []
	with torch.no_grad():

		for (data, label) in test_loader:   
			data, label = data.to(device).float(), label.to(device).long()

			output = model(data)
			predict = output.data.max(1)[1]

			for e in output.data.max(1)[1]:
				test_predict.append(e.item())
			for e in label:
				test_label.append(e.item())

	test_acc = accuracy_score(test_label, test_predict)
	print("test accuracy:", test_acc)
	Visualization.plot_confusion_matrix(test_label, test_predict, labels= None, \
				normalize = cm_normalize, cmap = plt.cm.Blues, title = None)