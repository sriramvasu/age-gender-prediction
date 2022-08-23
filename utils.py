import torch
import torch.nn as nn
# import torch.legacy.nn as lnn
import onnx
# from onnx_tf.backend import prepare
import onnxruntime
from functools import reduce
from torch.autograd import Variable
import numpy as np
import glob
import os
import random
import tensorflow as tf
import cv2
import albumentations as A
from sklearn.metrics import confusion_matrix
import datetime

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))

def pth2onnx(pth_path):
	vggface_model = nn.Sequential(  #Auto-generated using convert_torch_to_pytorch opensource package
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
	# nn.Softmax()
	)
	input_dir=pth_path.rsplit('/', 1)[0]
	vggface_model.load_state_dict(torch.load(pth_path))
	x=torch.randn(2, 3, 224, 224)
	face_descriptor=vggface_model(x)
	torch.onnx.export(vggface_model, x, os.path.join(input_dir, 'vggface.onnx'), export_params=True, input_names = ['input'], output_names = ['output'], dynamic_axes={'input':{0:'batch_size'}, 'output': {0:'batch_size'}})


def run_onnx(onnx_name='vggface.onnx'):
	session=onnxruntime.InferenceSession(onnx_name, None)
	input_name, output_name= session.get_inputs()[0].name, session.get_outputs()[0].name
	out=session.run([output_name], {input_name: np.random.randn(20,3,224,224).astype(np.float32) })
	print(out[0].shape)


def create_trainval_split(input_path):
	os.chdir(input_path)
	folders=[f for f in  glob.glob('*') if os.path.isdir(f)]
	images_list=[glob.glob(os.path.join(folder, '*/*')) for folder in folders]
	images_list=[j for i in images_list for j in i]
	random.shuffle(images_list)
	trainfile=open('train.txt', 'w')
	validfile=open('valid.txt','w')
	testfile=open('test.txt','w')
	trainfile.writelines([f+'\n' for f in images_list[:int(0.75*len(images_list))]])
	validfile.writelines([f+'\n' for f in images_list[int(0.75*len(images_list)):int(0.90*len(images_list))]])
	testfile.writelines([f+'\n' for f in images_list[int(0.90*len(images_list)):]])
	trainfile.close()
	validfile.close()
	testfile.close()