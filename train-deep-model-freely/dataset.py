import torch.utils.data as data
import os,sys
import numpy as np
from IPython import embed
import random
from PIL import Image
import torch



class DatasetFromFolder(data.Dataset):
    def __init__(self,split_id=1,mode='train',item='2d',input_transform=None,num_seg=3):
        #super(DatasetFromFolder,self).__init__()
        self.mode=mode
        self.root_dir='/S2/MI/zqj/video_classification/data/ucf101/'
        self.input_transform=input_transform
        self.s_id=split_id
        self.item=item
        self.num_seg=num_seg
        self.image_filenames,self.labels_list=self.get_images(os.path.join(self.root_dir,'metadata',mode+'_list0'+str(split_id)+'.txt'))
        self.training_size=224
        self.crop_ix=self.get_crop_ix(self.training_size,340,256)

    def get_crop_ix(self,training_size,mother_w,mother_h):
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
        return crop_ix

    def __getitem__(self,index):
        dir_files=self.image_filenames[index]
        len_files=len(os.listdir(dir_files))
        if self.item=='2d':
            num_frames=self.num_seg
            mat_data=torch.zeros(num_frames,3,224,224)
            if self.mode=='train':
                filenames=self.get_filenames_2d(dir_files,len_files,num_frames)
                rand_int=np.random.randint(0,10)
                for i,filename in enumerate(filenames):
                    img=Image.open(filename)
                    mat_data[i,:,:,:]=torch.from_numpy(self.transform_rgb_test(img,rand_int))
            else:# if mode=='test':
                filenames=self.get_filenames_2d_evenly(dir_files,len_files,num_frames)
                for i,filename in enumerate(filenames):
                    img=Image.open(filename)
                    mat_data[:,i,:,:,:]=torch.from_numpy(self.transform_rgb_test(img))

        else: #item=='3d'
            num_segements=self.num_seg
            mat_data=torch.zeros(10,16,num_segements,3,224,224)
            filenames=get_filenames_3d(dir_files,len_files,num_segements)

        target=self.labels_list[index]
        #del img
        return mat_data,target

    def __len__(self):
        return len(self.image_filenames)

    def transform_rgb_test(self,img,rand_int):
        mother_img=img.resize((340,256))
        crop_inds=self.crop_ix[rand_int/2]
        img_return=np.array(mother_img.crop(crop_inds),dtype=np.float32).transpose([2,0,1])
        if rand_int%2==0:
            img_return[:,:,:]=img_return[:,:,::-1]
        img_return[:,:,:]=img_return[::-1,:,:]
        img_return[0,:,:]-=104
        img_return[1,:,:]-=116
        img_return[2,:,:]-=122
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