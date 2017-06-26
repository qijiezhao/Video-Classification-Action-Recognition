import argparse
import os,sys,math,random
import torch,torchvision
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from models import inception_v4,VC_resnet101
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from IPython import embed
from data import get_training_set, get_testing_set
import argparse

parser = argparse.ArgumentParser(description="train the ucf101")
parser.add_argument('--model',type=str,default='resnet101',help='the training model(default as finetuning on ImageNet)')
parser.add_argument('--id',type=str,default=1,help='split id')
parser.add_argument('--batch_size',type=int,default=32,help='the batch size')
parser.add_argument('--gpu_id',type=int,default=0,help='the id of gpu that choose to train')
parser.add_argument('--num_workers',type=int, default=1,help='the num of workers that input training data')
parser.add_argument('--base_lr',type=float,default=0.01,help='the base learning rate')
parser.add_argument('--num_class',type=int,default=101,help='the number of classes')
parser.add_argument('--num_seg',type=int,default=3,help='number of segments')

args=parser.parse_args()

# traindir = os.path.join(args.data, 'train')
# valdir = os.path.join(args.data, 'val')
# train = datasets.ImageFolder(traindir, transform)
# val = datasets.ImageFolder(valdir, transform)
# train_loader = torch.utils.data.DataLoader(
#     train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)
def get_top1(label_sum,out_sum):
    label_sum=label_sum.numpy()
    out_sum=out_sum.numpy().argmax(1)
    assert len(label_sum)==len(out_sum)
    accuracy=sum([1 for i in range(len(label_sum)) if label_sum[i]==out_sum[i]])/float(len(label_sum))
    return accuracy

def exp_lr_scheduler(optimizer,epoch,init_lr=0.01,lr_decay_num0=1):
    lr=init_lr*(0.1**(epoch//lr_decay_num0))
    if epoch%lr_decay_num0==0:
        print 'lr is set to {}'.format(lr)
    # optimizer.param_groups[0]['lr']=lr*0.1
    # optimizer.param_groups[1]['lr']=lr
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    return optimizer

model=args.model
split_id=args.id
gpu_id=args.gpu_id
batch_size=args.batch_size
num_workers=args.num_workers
num_seg=args.num_seg

if torch.cuda.is_available():
    print 'gpu mode!'
    cuda=True
else:
    print 'cpu mode!'
    cuda=False

print 'loading dataset...'

train_set=get_training_set(split_id=split_id,num_seg=num_seg)
test_set=get_testing_set(split_id=split_id,num_seg=25)
training_data_loader=DataLoader(dataset=train_set,num_workers=num_workers,batch_size=batch_size,shuffle=True)
testing_data_loader=DataLoader(dataset=test_set,num_workers=num_workers,batch_size=10,shuffle=False)

base_lr=args.base_lr
num_class=args.num_class
if model=='resnet101':
    Net=VC_resnet101(num_class,gpu_id,num_seg).cuda()

#Net=inception_v4(num_class,gpu_id).cuda(gpu_id)
#Net=torch.load('../tmp_model/model_inceptionv4_epoch_0.pth').cuda(gpu_id)
#Net=torchvision.models.resnet50(pretrained=True).cuda()
loss_function=nn.CrossEntropyLoss().cuda(gpu_id)
#optimizer=optim.Adam(Net.parameters(),lr=0.01)
#embed()
#tunable_parameters=[{'params':Net.inception_v4.features[21].parameters(),'lr':0.1*base_lr},{'params':Net.inception_v4.classif.parameters()}]
optimizer=optim.SGD(Net.parameters(),lr=base_lr,momentum=0.9)#,weight_decay=0.00005)
Net2=torchvision.models.resnet101(pretrained=True).cuda()
Net2.fc=nn.Linear(2048,101).cuda()
avgpool2d=nn.AvgPool2d(kernel_size=(3,1)).cuda()
def train(epoch):
    #epoch_loss=0
    num0=0
    hit_ones=[]
    acc_gap=[]
    Net.train()
    label_sum,out_sum=torch.zeros(batch_size),torch.zeros(batch_size)
    for iteration,(inputss,labelss) in enumerate(training_data_loader,1):
        #embed()
        inputss,labelss=Variable(inputss).view(-1,3,224,224),Variable(labelss)
        len_data_train=len(labelss)
        if cuda:
            inputss=inputss.cuda(gpu_id)
            labelss=labelss.cuda(gpu_id)
        optimizer.zero_grad()
        #embed()
        output=Net(inputss).view(len_data_train,num_class)
        # x=Net2(inputss)
        # x=x.view(-1,1,num_seg,num_class)
        # output=avgpool2d(x).view(batch_size,num_class)

        loss=loss_function(output,labelss)
        loss.backward()
        optimizer.step()
        #embed()
        #epoch_loss+=loss
        if label_sum.sum()==0:
            label_sum=labelss.cpu().data
            out_sum=output.cpu().data
        else:
            label_sum=torch.cat([label_sum,labelss.cpu().data])
            out_sum=torch.cat([out_sum,output.cpu().data])
        if iteration%50==0:
            #embed()
            hit_one=get_top1(label_sum,out_sum)
            #hit_one=eval_util.calculate_hit_at_one(out_sum.numpy(),label_sum.numpy())
            print '===> Epoch[{}]({}/{}): loss:{:.4f}, hit_one_accuracy:{:.4f}'.format(epoch,iteration,7630,loss.data[0],hit_one)
            label_sum,out_sum=torch.zeros(batch_size),torch.zeros(batch_size)
        else:
            print '===> Epoch[{}]({}/{}): loss:{:.4f}'.format(epoch,iteration,9300,loss.data[0])
        del inputss,labelss


def test(epoch):
    Net.eval()
    preditions=[]
    for iteration,(inputss,labelss) in enumerate(testing_data_loader,1):
        inputss=Variable(inputss)
        if cuda:
            inputss=inputss.cuda(gpu_id)
        prediction=Net(inputss).cpu().data.numpy()
        prediction=prediction.sum(0).argmax()
        preditions.append(str(prediction))
        print 'video num: ',iteration,' predition: ',str(prediction)
    with open('/n/zqj/video_classification/data/ucf101/tmp_result/result_inceptionv4_new_epoch'+str(epoch)+'.txt','w')as fw:
        fw.write('\n'.join(preditions))

def checkpoint(epoch):
    model_out_path = "../tmp_model/model_resnet101_epoch_{}.pth".format(epoch)
    torch.save(Net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(0,6):
    optimizer=exp_lr_scheduler(optimizer,epoch,base_lr,2)
    train(epoch)
    checkpoint(epoch)
    #test(epoch)




