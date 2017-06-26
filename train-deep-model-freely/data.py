import os,sys
import numpy as np
from dataset import DatasetFromFolder

def get_training_set(split_id=1,num_seg=3):
    return DatasetFromFolder(split_id=split_id,mode='train',item='2d',num_seg=num_seg)

def get_validation_set(split_id,transform):
    return DatasetFromFolder(split_id=1,mode='val')

def get_testing_set(split_id=1,num_seg=25):
    return DatasetFromFolder(split_id=1,mode='test',item='2d',num_seg=num_seg)

