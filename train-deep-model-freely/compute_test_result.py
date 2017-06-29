import os,sys,numpy as np
from IPython import embed
def get_top1(gts,pres):
    len_items=len(gts)
    return sum([1 for i in range(len_items) if gts[i]==pres[i]])/float(len_items)

def get_confusion(gts,pres):
    confusion_mat=np.zeros([101,101])
    len_items=len(gts)
    for i in range(len_items):
        confusion_mat[gts[i]][pres[i]]+=1
    for i in range(101):
        confusion_mat[i,:]/=sum(confusion_mat[i,:])
    #embed()
    return confusion_mat

root='/S2/MI/zqj/video_classification/data/ucf101/'

root_meta=os.path.join(root,'metadata')
root_tmp=os.path.join(root,'tmp_result')
gt_file=os.path.join(root_meta,'classInd.txt')
class_names=[]
with open(gt_file,'r')as fp:
    lines=fp.readlines()
    for line in lines:
        class_names.append(line.strip().split(' ')[1])
file_name=sys.argv[1]
video_gt=os.path.join(root_meta,'testlist01.txt')
gt_labels=[]
with open(video_gt,'r')as fp:
    lines=fp.readlines()
    for line in lines:
        gt_labels.append(class_names.index(line.strip().split('/')[0]))

prediction_file=os.path.join(root_tmp,file_name)
predictions=[]
with open(prediction_file,'r')as fp:
    lines=fp.readlines()
    for line in lines:
        predictions.append(int(line.strip()))

top1_accuracy=get_top1(gt_labels,predictions)
print 'mean accuracy: ',top1_accuracy
# confusion_matrix=get_confusion(gt_labels,predictions)
# np.save(os.path.join(root,'tmp_result/confusionmat_result_lstm_hidden_epoch_6_01.npy'),confusion_matrix)