import torch
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable
import numpy as np
import glob
import os
import random
import tensorflow as tf
import cv2
import albumentations as A

class dataloader():
	def __init__(self, input_path='/home/sriram/Documents/gender_challenge/agegender_cleaned/combined', stage='test'):
		if(stage=='train'):
			self.imglist=[line.strip() for line in open(os.path.join(input_path, 'train.txt'), 'r').readlines()]
		if(stage=='valid'):
			self.imglist=[line.strip() for line in open(os.path.join(input_path, 'valid.txt'), 'r').readlines()]
		if(stage=='test'):
			self.imglist=[line.strip() for line in open(os.path.join(input_path, 'test.txt'), 'r').readlines()]
		self.idx=0
		self.input_path=input_path
		self.stage=stage
	def __iter__(self):
		return self
	def __next__(self):
		print(os.path.join(self.input_path, self.imglist[self.idx]))
		img=cv2.imread(os.path.join(self.input_path, self.imglist[self.idx]))
		label=self.imglist[self.idx].split('/')[1].split('_')[1]
		age=int(self.imglist[self.idx].split('/')[1].split('_')[0])
		self.idx+=1
		return img, label, age

def data_generator(stage='test', input_path='/home/sriram/Documents/gender_challenge/agegender_cleaned/combined'):
	if(stage=='train'):
		imglist=[line.strip() for line in open(os.path.join(input_path, 'train.txt'), 'r').readlines()]
	if(stage=='valid'):
		imglist=[line.strip() for line in open(os.path.join(input_path, 'valid.txt'), 'r').readlines()]
	if(stage=='test'):
		imglist=[line.strip() for line in open(os.path.join(input_path, 'test.txt'), 'r').readlines()]
	idx=0
	transforms=A.Compose([A.Resize(224,224), A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])
	train_pixeltransforms=A.Compose([
                            A.RandomContrast(limit=(-0.2,0.2), p=0.5),
                            A.RandomBrightness(limit=(-0.2, 0.2), p=0.5),
                            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
                            A.GaussNoise(var_limit=(10,30), p=0.5)])

	for idx in range(len(imglist)):
		img=cv2.imread(os.path.join(input_path, imglist[idx]))
		img=train_pixeltransforms(image=img)['image']
		img=transforms(image=img)['image']		
		label=imglist[idx].split('/')[1].split('_')[1]
		if(label=='M'):
			t=0
		elif(label=='F'):
			t=1
		age=int(imglist[idx].split('/')[1].split('_')[0])
		age_class=int(float(age)/5)
		if(age==70):
			age_class=13
		yield img.transpose([2,0,1]), np.int32(t), np.int32(age), np.int32(age_class)