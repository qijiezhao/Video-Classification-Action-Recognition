import os


params1=[0.1,0.5,1,5,20,100]
params2=[1,5,15,50,200,1000]

for param1 in params1:
    for param2 in params2:
        str_out=str_prefix='python svm_rbf.py --id=2 --data_type=global_mean --is_shuffle=True --is_subtrain=0.2 --svm_kernel=x2 --sigma={} --C={}'.format(param2,param1)
        print str_out
        os.system(str_out)

