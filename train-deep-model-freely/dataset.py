import torch.utils.data as data
import os,sys
import numpy as np
from IPython import embed
import random
from PIL import Image
import torch



class DatasetFromFolder(data.Dataset):
    def __init__(self,split_id=1,mode='train',item='2d',input_transform=None,num_seg=3,training_size=224):
        #super(DatasetFromFolder,self).__init__()
        self.mode=mode
        self.root_dir='/S2/MI/zqj/video_classification/data/ucf101/'
        self.input_transform=input_transform
        self.s_id=split_id
        self.item=item
        self.num_seg=num_seg
        self.image_filenames,self.labels_list=self.get_images(os.path.join(self.root_dir,'metadata',mode+'_list0'+str(split_id)+'.txt'))
        self.training_size=training_size
        self.rescale_size=self.init_scale(self.training_size)
        self.crop_ix=self.get_crop_ix(self.training_size)


    def init_scale(self,training_size):
        scales=[]
        w,h=training_size*(4/3.0),training_size
        fac=[float(256)/i for i in [256,224,192,168]]
        for i in range(4):
            scales.append((int(w*fac[i]),int(h*fac[i])))
        return scales

    def get_crop_ix(self,training_size):
        rescale_sizes=self.rescale_size
        crop_inds=[]
        for size_pair in rescale_sizes:
            mother_w,mother_h=size_pair
            crop_ix=np.zeros([5,4],dtype=np.int16)
            w_indices=(0,mother_w-training_size)
            h_indices=(0,mother_h-training_size)
            w_center=(mother_w-training_size)/2
            h_center=(mother_h-training_size)/2
            crop_ix[4,:]=[w_center,h_center,training_size+w_center,training_size+h_center]
            cnt=0
            for i in w_indices:
                for j in h_indices:
                    crop_ix[cnt,:]=[i,j,i+training_size,j+training_size]
                    cnt+=1
            crop_inds.append(crop_ix)
        return crop_inds

    def __getitem__(self,index):
        dir_files=self.image_filenames[index]
        len_files=len(os.listdir(dir_files))
        if self.item=='2d':
            num_frames=self.num_seg
            mat_data=torch.zeros(num_frames,3,self.training_size,self.training_size)
            if self.mode=='train':
                filenames=self.get_filenames_2d(dir_files,len_files,num_frames)
                rand_int=np.random.randint(0,20)
                for i,filename in enumerate(filenames):
                    img=Image.open(filename)
                    mat_data[i,:,:,:]=torch.from_numpy(self.transform_rgb_train(img,rand_int))
            else:# if mode=='test':
                filenames=self.get_filenames_2d_evenly(dir_files,len_files,num_frames)
                mat_data=torch.zeros(10,num_frames,3,self.training_size,self.training_size)
                for i,filename in enumerate(filenames):
                    img=Image.open(filename)
                    mat_data[:,i,:,:,:]=torch.from_numpy(self.transform_rgb_test(img,self.training_size))
        else: #item=='3d'
            num_segements=self.num_seg
            mat_data=torch.zeros(10,16,num_segements,3,self.training_size,self.training_size)
            filenames=get_filenames_3d(dir_files,len_files,num_segements)

        target=self.labels_list[index]
        #del img
        return mat_data,target

    def __len__(self):
        return len(self.image_filenames)

    def transform_rgb_train(self,img,rand_int):
        rescale_size_=self.rescale_size[rand_int%4]
        mother_img=img.resize(rescale_size_)
        crop_inds=self.crop_ix[rand_int%4][rand_int/4]
        img_return=np.array(mother_img.crop(crop_inds),dtype=np.float32).transpose([2,0,1])
        if rand_int%2==0:
            img_return[:,:,:]=img_return[:,:,::-1]
        img_return[:,:,:]=img_return[::-1,:,:]
        img_return[0,:,:]-=104
        img_return[1,:,:]-=116
        img_return[2,:,:]-=122
        #embed()
        return img_return

    def transform_rgb_test(self,img,train_size):
        mother_img=img # do not rescale in the testing process
        mother_w,mother_h=mother_img.size
        crop_ix=np.zeros([5,4],dtype=np.int16)
        w_indices=(0,mother_w-train_size)
        h_indices=(0,mother_h-train_size)
        w_center=(mother_w-train_size)/2
        h_center=(mother_h-train_size)/2
        crop_ix[4,:]=[w_center,h_center,train_size+w_center,train_size+h_center]
        cnt=0
        for i in w_indices:
            for j in h_indices:
                crop_ix[cnt,:]=[i,j,i+train_size,j+train_size]
                cnt+=1
        crop_ix=np.tile(crop_ix,(2,1))
        img_return=np.zeros([10,3,train_size,train_size])
        for i in range(10):
            cp=crop_ix[i]
            #embed()
            img_return[i]=np.array(mother_img.crop(cp),dtype=np.float32).transpose([2,0,1])  # transform w*h*channel to channel*w*h
        img_return[5:,:,:,:]=img_return[5:,:,:,::-1] #flipping
        img_return[:,:,:,:]=img_return[:,::-1,:,:]   #transform the RGB to BGR type
        img_return[:,0,:,:]-=104
        img_return[:,1,:,:]-=116
        img_return[:,2,:,:]-=122
            #embed()
        return img_return

    def get_filenames_2d(self,dir,len_files,num_out):
        each_=len_files/num_out
        if each_==0:
            each_=1
        inds=[]
        for i in range(num_out):
            s_=each_*i
            e_=each_*(i+1)
            inds.append(np.random.randint(s_,e_)+1)
        def rename(ind):
            len_lack=4-len(str(ind))
            return '0'*len_lack+str(ind)+'.jpg'
        names=[os.path.join(dir,'img_'+rename(ind)) for ind in inds]
        return names
    def get_filenames_2d_evenly(self,dir,len_files,num_out):
        each_=len_files/num_out
        inds=[each_*i+1 for i in range(num_out)]
        def rename(ind):
            len_lack=4-len(str(ind))
            return '0'*len_lack+str(ind)+'.jpg'
        names=[os.path.join(dir,'img_'+rename(ind)) for ind in inds]
        return  names

    def get_images(self,path):
        with open(path,'r')as fp:
            lines=fp.readlines()
            images=[]
            labels=[]
            for line in lines:
                img_path=os.path.join(self.root_dir,'imgs',line.strip().split('.')[0].split('/')[-1])
                images.append(img_path)
                labels.append(int(line.strip().split(' ')[1])-1)

        return images,labels