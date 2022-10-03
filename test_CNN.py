import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.optim import lr_scheduler
# import matplotlib.pyplot as plt
import argparse
import json
import os
import cv2
# import tikzplotlib
from PIL import Image
from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
# from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset_test
from config import configs
from pathlib import Path

def main():
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	dataset = args.save_name
	output_file = args.output_file
	model_name = args.model_name
	torch.backends.cudnn.benchmark = True
	config = configs[model_name]
	test_dataset = MyDataset_test(txt_path=test_list, transform=config.transform['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
	test_dataset_size = len(test_dataset)
	scaler = MinMaxScaler()
	true_negative = 0
	true_positive = 0
	false_negative = 0
	false_positive = 0
	y_test = []
	y_pred = []
	acc = 0
	#model = torchvision.models.densenet121(num_classes=2)
	# model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	# model = MesoInception4()
	model = config.model
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model = nn.DataParallel(model)
	model.eval()
	test_output_file = os.path.join('test_output', model_name, output_file)
	test_output_folder = Path(test_output_file).parent
	if not os.path.exists(test_output_folder):
		os.makedirs(test_output_folder)
	result_file = open(test_output_file, 'a')
	result_file.write('image_path\tprediction_score\toutput_label\treal_label\n')
	with torch.no_grad():
		for (image, labels, image_name) in test_loader:
			image = image.cuda()
			labels = labels.cuda()
			outputs, _ = model(image)
			# print(outputs)

			# Vit16
			if model_name == 'vit-16':
				outputs = outputs.logit

			result_file.write(image_name[0] + '\t')

			softmax = torch.nn.Softmax(dim=1)
			output_softmax = softmax(outputs)
			output1 = output_softmax[0][0].item()
			output2 = output_softmax[0][1].item()
			# output1_str = str(output1)
			output2_str = str(output2)
			result_file.write(output2_str + '\t')

			_, class_label = torch.max(outputs.data, 1)
			labels_scores_predictions = '{0}\t{1}'.format(class_label.cpu().item(), labels.cpu().item())
			result_file.write(labels_scores_predictions + '\n')

			_, preds = torch.max(outputs.data, 1)

			for i in range(len(preds)):
				if preds[i] == 0 and labels.data[i] == 0:
					true_negative += 1
				elif preds[i] == 1 and labels.data[i] == 1:
					true_positive += 1
				elif preds[i] == 1 and labels.data[i] == 0:
					false_positive += 1
				elif preds[i] == 0 and labels.data[i] == 1:
					false_negative += 1

			# y_test = y_test + true_labels
			# y_pred = y_pred + output_
			# y_test.extend(true_labels)
			# print(y_test)
			# y_pred.extend(output_)
			# print(y_pred)

		# if 'Deepfakes' in test_list:
		# 	dataset = 'mesoinc4_fs_df_c23'
		# elif 'Face2Face' in test_list:
		# 	dataset = 'mesoinc4_fs_f2f_c23'
		# elif 'FaceSwap' in test_list:
		# 	dataset = 'mesoinc4_fs_fs_c23'
		# elif 'NeuralTextures' in test_list:
		# 	dataset = 'mesoinc4_fs_nt_c23'

		# print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		acc = (true_positive + true_negative) / (true_negative + false_negative + false_positive + true_positive)
		# with open('plots/' + dataset + '_labels.txt', 'w') as f:
		# 	json.dump([int(i) for i in y_test], f)
		# with open('plots/' + dataset + '_prediction.txt', 'w') as f:
		# 	json.dump(y_pred, f)
		# roc_curve
		# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		# auc_value = auc(false_positive_rate, true_positive_rate)

		print('true negative: {}, false negative: {}, false positive: {}, true positive: {}'.format(
			true_negative, false_negative, false_positive, true_positive))
		print('Test Acc: {:.4f}'.format(acc))
		# print('AUC score: {:.4f}'.format(auc_value))

		# plt.title('ROC curve')
		# plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.3f' % auc_value)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'm--')
		# plt.xlim([0, 1])
		# plt.ylim([0, 1.1])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# tikzplotlib.save('plots/' + dataset + '.txt')
		# plt.savefig('plots/' + dataset + '.png')
	result_file.close()
	# print('true_negative: {}'.format(true_negative))
	# print('false_positive: {}'.format(false_positive))
	# print('false_negative: {}'.format(false_negative))
	# print('true_positive: {}'.format(true_positive))

if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--save_name', type=str, default='none')
	parse.add_argument('--model_name', type=str, default='wrong')
	parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
	parse.add_argument('--output_file', type=str, default='./output/test.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
	main()
	print('Hello world!!!')
