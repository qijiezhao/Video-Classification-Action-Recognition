import os,sys
import numpy as np
from sklearn import svm
import argparse,random
import pickle
from IPython import embed
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import preprocessing
import xgboost as xgb

ucf_data='/S2/MI/zqj/video_classification/data/ucf101/'
metadata_root=os.path.join(ucf_data,'metadata')
feature_root=os.path.join(ucf_data,'feature_rgb_pool')

def get_data(item='train',id=1,is_shuffle=False,is_subtrain=1):
    file_path=os.path.join(metadata_root,item+'_list0'+id+'.txt')
    files=[]
    labels=[]
    with open(file_path,'r')as fp:
        lines=fp.readlines()
        if is_shuffle==True:
            np.random.shuffle(lines)
        if not is_subtrain==1:
            lines=random.sample(lines,int(len(lines)*is_subtrain))
        for line in lines:
            tmp_prefix=line.strip().split('.')[0].split('/')[1]
            label_tmp=line.strip().split(' ')[1]
            files.append(os.path.join(feature_root,tmp_prefix+'.npy'))
            labels.append(int(label_tmp)-1)
    return files,np.array(labels,dtype=np.float64)

def get_mean(ndarray):
    a1=ndarray[:8].mean(0)
    a2=ndarray[8:17].mean(0)
    a3=ndarray[17:].mean(0)
    return np.hstack([a1,a2,a3])

def get_max(ndarray):
    a1=ndarray[:8].max(0)
    a2=ndarray[8:17].max(0)
    a3=ndarray[17:].max(0)
    return np.hstack([a1,a2,a3])

def get_global_mean(ndarray):
    return ndarray.mean(0)

def get_accuracy(pre,labels):
    assert len(pre)==len(labels)
    len_pre=len(pre)
    right=sum([1 for i in range(len_pre) if pre[i]==labels[i]])
    return right/float(len_pre)

def main():
    '''
    Note:
    the model which extracts these features should be trained on the same id split data.
    '''
    parser = argparse.ArgumentParser(description="perform SVM")
    parser.add_argument('--id',type=str,default=1,help='split id')
    parser.add_argument('--sigma',type=float,default=10,help='svm kernel\'s scale parameter, or lr for gbdt')
    parser.add_argument('--data_type',type=str,default='max',help='the aggregation type')
    parser.add_argument('--is_shuffle',type=bool,default=True,help='whether to shuffle the training data')
    parser.add_argument('--is_subtrain',type=float,default=1,help='whether to choose a subset of training data to perform short time training')
    parser.add_argument('--svm_kernel',type=str,default='linear',help='svm kernel,options: ')
    parser.add_argument('--C',type=float,default=1,help='svm parameter: C or iter-times for gbdt')
    args=parser.parse_args()

    id=args.id
    sigma=args.sigma
    data_type=args.data_type
    is_shuffle=args.is_shuffle
    is_subtrain=args.is_subtrain
    svm_kernel=args.svm_kernel
    penalty_C=args.C

    times=0
    '''
    three types to reshape the 25*1024 matrix (to 1*1024 or 1*3072)
    '''
    if data_type=='mean':
        Reshape_data=get_mean
        times=3
    elif data_type=='max':
        Reshape_data=get_max
        times=3
    elif data_type=='global_mean':
        Reshape_data=get_global_mean
        times=1
    else:
        raise Exception('wrong datatype')
    print 'reading data...'
    train_files,train_label=get_data(item='train',id=id,is_shuffle=is_shuffle,is_subtrain=is_subtrain)
    test_files,test_label=get_data(item='test',id=id,is_shuffle=False,is_subtrain=1)
    len_train,len_test=len(train_label),len(test_label)
    train_data=np.zeros([len_train,1024*times])
    test_data=np.zeros([len_test,1024*times])

    print 'read data done!'
    print 'transforming data...'


    for num,train_file in enumerate(train_files):
        train_data[num]=Reshape_data(np.load(train_file))
    for num,test_file in enumerate(test_files):
        test_data[num]=Reshape_data(np.load(test_file))

    print 'transform data!'
    print 'begin to train...'

    _svm=None
    if svm_kernel=='rbf':
        _svm=svm.SVC(C=penalty_C,kernel='rbf',probability=True)#,probability=True,gamma=1/sigma)
        _svm.fit(train_data,train_label)
    elif svm_kernel=='linear':
        _svm=svm.SVC(C=penalty_C,kernel='linear',probability=True)
        _svm.fit(train_data,train_label)
    elif svm_kernel=='sigmoid':
        _svm=svm.SVC(C=penalty_C,kernel='sigmoid',probability=True)
        _svm.fit(train_data,train_label)
    elif svm_kernel=='gbdt':
        '''
        initialization the xgboost model
        '''
        param = {
        'objective': 'multi:softprob',
        'eta': sigma,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        'nthread': 32,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': 101,
    }
        train = xgb.DMatrix(train_data, label=train_label)
        test=xgb.DMatrix(test_data,label=test_label)
        evallist  = [(train,'train'), (test,'eval')]
        model = xgb.train(param, train, int(penalty_C), evals=evallist, early_stopping_rounds=20)

    elif svm_kernel=='x2':
        '''
        pre-process the data
        '''
        train_data-=train_data.min()
        test_data-=test_data.min()
        train_data=preprocessing.normalize(train_data,norm='l1')
        test_data=preprocessing.normalize(test_data,norm='l1')
        _svm=svm.SVC(C=penalty_C,kernel=chi2_kernel,gamma=1/sigma,probability=True)
        #K=chi2_kernel(train_data,gamma=0.5)
        _svm.fit(train_data,train_label)
    else:
        raise Exception('not added yet')

    '''
    save the model
    '''
    if svm_kernel=='gbdt':
        save_file=os.path.join(ucf_data,'tmp_result','gbdt0'+id+'.pickle')
        pickle.dump(model, open(save_file, "wb"))
    else:
        save_file=os.path.join(ucf_data,'tmp_result','svm0'+id+'_sigma'+str(sigma)+'_C'+str(penalty_C)+'_kernel'+svm_kernel+'.pickle')
        pickle.dump(_svm,open(save_file,'wb'))

    print 'train done!'
    print 'begin to test...'
    if svm_kernel=='gbdt':
        pre_test=model.predict(test)
    else:
        pre_test=_svm.predict(test_data)
    #embed()
    accuracy=get_accuracy(pre_test,test_label)
    #embed()
    print 'accuracy:', accuracy

if __name__=='__main__':
    main()