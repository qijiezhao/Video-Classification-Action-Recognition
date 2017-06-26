from math import log10
import numpy as np
import torch,torchvision,os,sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from model import Net
from IPython import embed
import torch.nn.functional as F
from data import get_training_set, get_testing_set
import random
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt

'''
this file contains several models including:
ResNet_50, VGG19, inception_v4, C3D_version1, LSTM_version1

'''

def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2)*weights.size(3)*weights.size(4)

    u, _, v = svd(normal(0.0, 0.01, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))

class VC_resnet50(nn.Module):
    def __init__(self,out_size):
        super(VC_resnet50,self).__init__()
        self.softmax=nn.Softmax()
        self.resnet50=torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc=nn.Linear(2048,out_size).cuda()

    def forward(self,x):
        x=self.resnet50(x)
        return x#self.softmax(x)

class VC_vgg19(nn.Module):
    def __init__(self,out_size):
        super(VC_vgg19,self).__init__()
        self.softmax=nn.Softmax()
        self.vgg19=torchvision.models.vgg19(pretrained=True).cuda()
        mod = list(self.vgg19.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, out_size).cuda())
        new_classifier = torch.nn.Sequential(*mod)
        self.vgg19.classifier = new_classifier

    def forward(self,x):
        x=self.vgg19(x)
        return x
class VC_resnet101(nn.Module):
    def __init__(self,out_size,gpu_id,num_seg):
        super(VC_resnet101,self).__init__()
        self.gpu_id=gpu_id
        self.num_seg=num_seg
        self.resnet101=torchvision.models.resnet101(pretrained=True)#.cuda(self.gpu_id)
        mod=[nn.Dropout(p=0.8)]#.cuda(self.gpu_id)]
        mod.append(nn.Linear(2048,101))#.cuda(self.gpu_id))
        new_fc=nn.Sequential(*mod)#.cuda(self.gpu_id)
        self.resnet101.fc=new_fc
        #self.resnet101.fc=nn.Linear(2048,101).cuda(gpu_id)

        self.avg_pool2d=nn.AvgPool2d(kernel_size=(3,1))#.cuda(self.gpu_id)
    def forward(self,x):
        x=self.resnet101(x)
        x=x.view(-1,1,self.num_seg,101)#.cuda(self.gpu_id)
        x=self.avg_pool2d(x)
        return x

class inception_v4(nn.Module):
    def __init__(self,out_size,gpu_id):
        super(inception_v4,self).__init__()
        sys.path.insert(0,'../tool/models_zoo/')
        from inceptionv4.pytorch_load import inceptionv4
        self.inception_v4=inceptionv4(pretrained=True).cuda(gpu_id)
        # for params in self.inception_v4.parameters():
        #     params.requires_grad=False
        # for params in self.inception_v4.features[21].parameters():
        #     params.requires_grad=True
        self.inception_v4.classif=nn.Linear(1536,101).cuda(gpu_id)

    def forward(self,x):
        x=self.inception_v4(x)
        return x

class cnn_m(nn.Module):
    def __init__(self):
        super(cnn_m,self).__init__()
        self.conv1=nn.Conv2d(20,96,kernel_size=(7,7),stride=(2,2))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2=nn.Conv2d(96,256,kernel_size=(5,5),stride=(2,2))
        self.conv3=nn.Conv2d(256,512,kernel_size=(3,3),stride=1)
        self.conv4=nn.Conv2d(512,512,kernel_size=(3,3),stride=1)
        self.conv5=nn.Conv2d(512,512,kernel_size=(3,3),stride=1)
        self.fc6=nn.Linear(4608,4096)
        self.fc7=nn.Linear(4096,2048)
        self.fc8=nn.Linear(2048,101)
        self.norm=nn.BatchNorm2d(96)

    def forward(self,x):
        x=self.pool(self.norm(self.conv1(x)))
        x=self.pool(self.conv2(x))
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.pool(self.conv5(x)).view(-1,4608)
        x=self.fc8(self.fc7(self.fc6(x)))
        return x



class C3D_net(nn.Module):
    def __init__(self):
        super(C3D_net,self).__init__()
        self.conv1=nn.Conv3d(3,64,kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        self.relu=nn.ReLU()
        self.maxpool1=nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        self.conv2=nn.Conv3d(64,128,kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        self.maxpool2=nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.conv3=nn.Conv3d(128,256,kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        self.maxpool3=nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.conv4=nn.Conv3d(256,256,kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        self.maxpool4=nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.conv5=nn.Conv3d(256,256,kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        self.maxpool5=nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.num_out_maxpool5=2304
        self.fc6=nn.Linear(self.num_out_maxpool5,2048)#TBA
        self.fc7=nn.Linear(2048,2048)
        #self.dropout=nn.Dropout(p=0.5)
        self.fc8=nn.Linear(2048,101)
        self._initialize_weights()

    def forward(self,x):
        x=self.maxpool1(self.relu(self.conv1(x)))
        x=self.maxpool2(self.relu(self.conv2(x)))
        x=self.maxpool3(self.relu(self.conv3(x)))
        x=self.maxpool4(self.relu(self.conv4(x)))
        x=self.maxpool5(self.relu(self.conv5(x)))
        x=x.view(-1,self.num_flat_features(x))
        x=self.relu(self.fc6(x))
        x=F.dropout(x,training=self.training)
        x=self.relu(self.fc7(x))
        x=F.dropout(x,training=self.training)
        x=self.fc8(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv3.weight.data.copy_(_get_orthogonal_init_weights(self.conv3.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))
        self.conv5.weight.data.copy_(_get_orthogonal_init_weights(self.conv5.weight))

class LSTM_ver1(nn.Module):
    def __init__(self,batch_size,input_size,hidden_size,num_layers,dropout=0,bidirectional=False):
        super(LSTM_ver1,self).__init__()
        self.drop=nn.Dropout(0.5)
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=bidirectional,batch_first=True)
        self.hidden_data=self.hidden_initial(num_layers,batch_size, hidden_size)
        self.fc=nn.Linear(hidden_size,101)
    def forward(self,x):
        #minibatch_size,seq_len,feature_dic=x.size()
        #embed()
        out,hidden_n=self.lstm(x,self.hidden_data)
        #embed()
        return self.fc(hidden_n[0][1,:,:])

    def hidden_initial(self,num_layers,batch_size,hidden_dim):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(num_layers, batch_size, hidden_dim)),
                Variable(torch.zeros(num_layers, batch_size, hidden_dim)))
