import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2

from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms, xception_aug_transforms, mesonet_transforms
from dataset.mydataset import MyDataset
from config import configs


# For weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main():
	torch.cuda.empty_cache()
	args = parse.parse_args()
	name = args.name
	continue_train = args.continue_train
	print(continue_train)
	train_list = args.train_list
	val_list = args.val_list
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('output', name)
	if not os.path.exists(output_path):
		os.mkdir(output_path)

	torch.backends.cudnn.benchmark = True
	config = configs[model_name]
	# whole_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
	# whole_dataset = MyDataset_wavelet(txt_path=train_list, transform=xception_default_data_transforms['train'])
	# val_dataset = MyDataset_wavelet(txt_path=val_list, transform=xception_aug_transforms['val'])

	# train_dataset_lenght = int(0.7*len(whole_dataset))
	# val_dataset_length = len(whole_dataset) - train_dataset_lenght

	# train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
	# val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])

	train_dataset = MyDataset(txt_path=train_list, transform=config.transform['train'])
	val_dataset = MyDataset(txt_path=val_list, transform=config.transform['val'])

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

	train_dataset_size = len(train_dataset)
	val_dataset_size = len(val_dataset)
	# model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	# model = MesoInception4()
	model = config.model
	if continue_train:
		model.load_state_dict(torch.load(model_path))
	model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	# for xception
	# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
	# optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, cooldown=0, min_lr=1e-8, verbose=True)
	model = nn.DataParallel(model)
	# model.apply(weights_init)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0
	for epoch in range(epoches):
		with open("result.txt", 'a') as output:
			output.write('Epoch {}/{}'.format(epoch+1, epoches) + '\n')
			output.write('-'*10 + '\n')
		print('Epoch {}/{}'.format(epoch+1, epoches))
		print('-'*10)
		model.train()
		train_loss = 0.0
		train_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0
		for (image, labels) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			image = image.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(image)

			# Vit16
			if model_name == 'vit-16':
				outputs = outputs.logits

			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			iter_loss = loss.data.item()
			train_loss += iter_loss
			for i in range(len(preds)):
				if preds[i] == labels.data[i]:
					iter_corrects += 1
			train_corrects += iter_corrects
			iteration += 1
			if not (iteration % 100):
				with open("result.txt", 'a') as output:
					output.write('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size) + '\n')
				print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size
		with open("result.txt", 'a') as output:
			output.write('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
		print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

		model.eval()
		with torch.no_grad():
			for (image, labels) in val_loader:
				image = image.cuda()
				labels = labels.cuda()
				outputs = model(image)

				# Vit16
				if model_name == 'vit-16':
					outputs = outputs.logits

				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				val_loss += loss.data.item()
				for i in range(len(preds)):
					if preds[i] == labels.data[i]:
						val_corrects += 1
			epoch_loss = val_loss / val_dataset_size
			epoch_acc = val_corrects / val_dataset_size
			print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()

		scheduler.step()
		# scheduler.step(val_loss)
		print('---------latest learning rate---------')
		print(optimizer.param_groups[0]['lr'])
		# if not (epoch % 40):
		torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
	with open("result.txt", 'a') as output:
		output.write('Best val Acc: {:.4f}'.format(best_acc) + '\n')
	print('Best val Acc: {:.4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))




if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='fs_xception_c0_299')
	parse.add_argument('--train_list', '-tl', type=str, default='')
	parse.add_argument('--val_list', '-vl', type=str, default='')
	parse.add_argument('--batch_size', '-bz', type=int, default=64)
	parse.add_argument('--epoches', '-e', type=int, default='5')
	parse.add_argument('--model_name', '-mn', type=str, default='fs_c0_299.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
	main()
