import os,sys
import numpy as np
from IPython import embed
root='/n/zqj/video_classification/data/'
root_data=os.path.join(root,'ucf101')
root_meta=os.path.join(root_data,'metadata')
train_list=os.path.join(root_meta,'trainlist01.txt')
test_list=os.path.join(root_meta,'testlist01.txt')

def choose(files):
    len_frames=len(files)/3
    len_frames=(len_frames/10)*10
    def add_0(num):
        str_0='0'*(4-len(str(num)))
        return str_0+str(num)
    flow_x=['flow_x_'+add_0(i+1)+'.jpg' for i in range(len_frames)]
    flow_y=['flow_y_'+add_0(i+1)+'.jpg' for i in range(len_frames)]
    return_str=[]
    for i in range(len_frames/10):
        s=i*10
        e=(i+1)*10
        return_str.extend(flow_x[s:e])
        return_str.extend(flow_y[s:e])
    return return_str

with open(train_list,'r')as fp:
    lines_train=fp.readlines()
    out_train=[]
    for line in lines_train:
        frame_path=os.path.join(root_data,'opticalflow',line.strip().split(' ')[0].split('.')[0])
        files=os.listdir(frame_path)
        files=choose(files)
        for file in files:
            full_path=os.path.join(frame_path,file)
            out_train.append(full_path+' '+str(int(line.strip().split(' ')[1])-1))
with open(os.path.join(root_meta,'train1_opticalflows_list.txt'),'w')as fw:
    fw.write('\n'.join(out_train))

print 'train done!'

with open(test_list,'r')as fp:
    lines_test=fp.readlines()
    out_test=[]
    for line in lines_test:
        frame_path=os.path.join(root_data,'frames',line.strip().split(' ')[0].split('.')[0])
        files=os.listdir(frame_path)
        files=choose(files)
        for file in files:
            full_path=os.path.join(frame_path,file)
            out_test.append(full_path)
with open(os.path.join(root_meta,'test1_opticalflows_list.txt'),'w')as fw:
    fw.write('\n'.join(out_test))

print 'test done!'