import os,sys
import numpy as np
root='/n/zqj/video_classification/data/'
root_data=os.path.join(root,'ucf101')
root_meta=os.path.join(root_data,'metadata')
train_list=os.path.join(root_meta,'trainlist01.txt')
test_list=os.path.join(root_meta,'testlist01.txt')

with open(train_list,'r')as fp:
    lines_train=fp.readlines()
    out_train=[]
    for line in lines_train:
        frame_path=os.path.join(root_data,'frames',line.strip().split(' ')[0].split('.')[0])
        files=np.array(os.listdir(frame_path))
        len_files=len(files)
        each_=len_files/40
        inds_files=np.array([i*each_ for i in range(40)])
        files=files[inds_files]
        for file in list(files):
            full_path=os.path.join(frame_path,file)
            out_train.append(full_path+' '+str(int(line.strip().split(' ')[1])-1))
with open(os.path.join(root_meta,'train1_frames_list.txt'),'w')as fw:
    fw.write('\n'.join(out_train))

print 'train done!'

with open(test_list,'r')as fp:
    lines_test=fp.readlines()
    out_test=[]
    for line in lines_test:
        frame_path=os.path.join(root_data,'frames',line.strip().split(' ')[0].split('.')[0])
        files=np.array(os.listdir(frame_path))
        len_files=len(files)
        each_=len_files/40
        inds_files=np.array([i*each_ for i in range(40)])
        files=files[inds_files]
        for file in list(files):
            full_path=os.path.join(frame_path,file)
            out_test.append(full_path)
with open(os.path.join(root_meta,'test1_frames_list.txt'),'w')as fw:
    fw.write('\n'.join(out_test))

print 'test done!'
