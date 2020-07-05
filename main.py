import data_loader
import PSNet

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def adjust_learning_rate(optimizer, weight_decay = 0.9):

	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * weight_decay

def my_collate(batch):

	data = [torch.FloatTensor(item[0]).to(device) for item in batch]
	target = [int(item[1]) for item in batch]
	target = torch.LongTensor(target)
	return [data, target]

if __name__ == '__main__':

	version = '1'
	Batch_size = 32
	lr = 0.001
	epochs = 50
	model = PSNet.PSNET()
	optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999))

	train_loader = Data.DataLoader(data_loader.densepointdataset(mode = 'train'),\
	 batch_size = Batch_size, shuffle = True)#, collate_fn = my_collate)
	
	test_loader = Data.DataLoader(data_loader.densepointdataset(mode = 'test'), \
	 batch_size = Batch_size, shuffle = True)#, collate_fn = my_collate)

	print(f'learning rate: {lr}, Batch size: {Batch_size}, epochs: {epochs}\n')
	model.to(device)
	
	Loss = nn.CrossEntropyLoss()

	max_test_acc = 0
	#start training
	for epoch in range(1,epochs+1):
		print("epoch: ", epoch)
		train_label = []
		train_predict = []
		#train
		model.train()
		for batch, (data, label) in enumerate(train_loader, 1):
			data, label = data.to(device).float(), label.to(device).long()
			#label = label.to(device)
			optimizer.zero_grad()
			output = model(data)
			predict = output.data.max(1)[1]
			
			for e in output.data.max(1)[1]:
				train_predict.append(e.item())
			for e in label:
				train_label.append(e.item())

			loss = Loss(output, label)

			loss.backward()
			optimizer.step()

			if batch %100 == 0:
				print(f'now batch: {batch}')

		#test
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
		max_test_acc = max(max_test_acc, test_acc)

		if epoch % 5 == 0:
			torch.save(model, './save_model/PSNet_' + str(epoch) + '_' + version)