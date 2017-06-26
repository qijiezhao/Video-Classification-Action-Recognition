### 北京大学研究生课程：
# 计算视觉高级专题 - course project
-----
Author: Qijie Zhao
Date: 06/26/2017


----------

### 任务：

视频分类

### 任务描述：

对短时序视频进行视频分类。通过对训练集视频进行训练，使模型具有良好的视频分类能力，在测试阶段，模型需要给测试集中每个视频一个类别标签。

### 数据集：

UCF101（full）#,hmdb51(testing)

### 评测标准：

average accuracy （在3个split上）

----

----------

### 实验介绍：

本次任务，实验尝试了C3D[4], TSN[2], DOVF[3], TS_LSTM[1](code testing)等思路。


----------

#### (0)预处理：

首先要设置好实验数据的路径和caffe等依赖外部文件的路径（略）。


path: data-preprocess/

	python data-preprocess/generate_traintest_list.py #生成训练集和测试集的list
	python data-preprocess/generate_optical_flow_traintest_list.py #生成光流的训练集和测试集的list
	python data-preprocess/build_of.py (source_path) (out_path) --df_path=(denseflow_path)#生成光流,需要指定一些路径


#### (1)C3D:

path: c3d-part/

本实验是基于C3D 1.1 version, 分为3部分：

##### a) train from scratch
	cd c3d-part/c3d_ucf101_training
	sh train_simple.sh
##### b) finetune from pretrained model
	cd c3d-part/c3d_ucf101_finetuning
	sh finetuing_ucf101.sh
##### c) extract features using the pretrained model
	cd c3d-part/c3d_feature_extraction
	sh feature_extraction.sh

##### 实验结果：

1，train from scratch 如果只用ucf101训练，一直不收敛，可能需要一些新的大些的数据集pretrain才能收敛。

2，training from pretrained model，pretrained model是在sports1m上面训练的，基础模型是resnet18。 收敛比较快，batch_size为8，训练40000次就能到测试集精度81%。 

3，用训练好的C3D抽取的feature，然后用linear-SVM训练并分类，在测试集上的精度能到82.3%，使用chi2-svm训练并分类能达到85.3%的精度。文中的90%左右的精度是通过和iDT[5]做融合得到的结果(x)。


#### (2)TSN:

path: train-model+extract-feature/

本实验是基于TSN的，实验过程使用了一些TSN的code。

同样包括(a)训练模型和(b)用模型抽取图像帧数据的feature：

模型文件在 train-model+extract-feature/models下

##### 训练模型：
	
	sh train-model+extract-feature/train_tsn.sh

##### 用模型抽取feature：

	python train-model+extract-feature/extrac_features.py ucf101 1 rgb (frame_path) (net_pro) (net_weights)

（需要抽取的层在代码中的参数score_name）

##### 用训练的模型测试结果：

	python train-model+extract-feature/eval_net.py ucf101 1 rgb (frame_path) (net_pro) (net_weights)

#### 实验结果：

1，TSN的spatial frame部分得到了和论文中一样的结果，86%左右。光流部分，暂时使用的opencv::calcOpticalFlowFarneback()计算光流，效果不如TVL1，只有79%（end-to-end 的测试集结果）

2，TSN在训练单模型方面，效果是目前最好的，主要有如下几点原因(后来很多文章都沿用了这些trick)：

a), Data augmentation （training）： 4 scales x (4 corners + 1 center) x 2 flipped ,输入数据有多个crop的方式，这使得输入训练的数据具备多尺度，多角度，以及对称不变等特点。 同时也让数据集“变”大了。

b),同一个视频一次只送入n张训练（一个视频被切分成n个segments，实验里n=3），然后分别在每个segment里面随机挑选一张，这n张图像在通过网络之后，其结果会再过一个average-pooling, 这使得不至于太多相似图像一起训练造成网络很容易过拟合。

c),在测试阶段，TSN把一个视频平均取25个frame，每个frame经过 （4 corner + 1 center) * 2 flipped 的crop操作，然后把这10张图像取一个平均值。

3，文中使用的bn_inception结构虽然在图像分类任务上不及resnet等模型，但是却能很好的适用于数据量不大，容易更拟合的任务，比如像ucf101一样的视频分类数据集。

#### (3)DOVF:

path: svm+gbdt/

为了方便，我把gbdt也放在了svm_kernel的选择里面，实际上gbdt和svm是不相干的。

仿照[3]的实验设计，我用TSN的网络抽取了ucf101所以视频的frame-level的feature，方法是按照测试集的处理，即平均取25个frame，然后每个frame做10份crop，并在输出的结果里面将这10个结果做average-pooling，之后得到25个feature。层是选择的第一层fc层，即导数第二层，这在[3]中被证明是bn_inception中表达能力最强的一层，同时包含足够的spatial information和high-level semantic information。然后这些feature通过三种不同的方式被送入SVM中进行训练：

a)25个feature直接取average-pooling

b)8+9+8的concatenation的方式，其中各小段内部使用average-pooling

c)8+9+8的concatenation的方式，其中各小段内部使用max-pooling

实验证明，c)效果最好，在split01的测试集上能达到88%

gbdt对于本任务效果不太好，通过一系列调参，最多只能把测试集精度做到80.3%

code implementation:

	python svm+gbdt/svm_classification.py --id=1 --sigma=1 --data_type=max --is_shuffle=True --is_subtrain=1 --svm_kernel=linear/x2/rbf/gbdt --C=1 


#### (4)TS_LSTM:

path: train-deep-model-freely/

这是最后的尝试，也是正在继续做的部分。

首先，这篇文章在resnet101上面训练得到了很接近bn_inception的结果，他的data augmentation的步骤学习了TSN的思路。 其次，这篇文章也是在模型抽取的feature（spatial concated with flow）上面做pooling的实验，分别尝试了LSTM和 conv-pooling，之前我在youtube-8m上面也做过相关的尝试，但是都不work，这篇文章虽然没有超过state of the art, 不过也几乎接近95%，这个工作可以继续follow下去（先用更灵活的代码复现出来），并把attention-based LSTM, MOE-RNN等也用到视频分类中，可以换一些更适合的数据集。

code implementation:

	python train-deep-model-freely/main.py --model=resnet101 --id=1 --batch_size=25 --gpu_id=0 --num_workers=4 --base_lr=0.01 --num_classes=101 --num_seg=3

目前复现的工作在训练集上97%（spatial level）,剩下的工作还在继续。

-----

### *未来工作


检测是在图像中标出bounding box，这是图像中的high-level semantic information。 而对于视频，则应该是一个cube，来做同样的事情。 如果视频分类还有下一步提升空间，很可能就在这个方向。 在图像中，训练好的模型，根据高层的激活单元反向传播，能看到激活这个神经元所需要的原图pixel，这表明是这些像素表达了该高层中该结点的激活。视频中同样应该有一个表示多帧中做activity的cube/continuous-part。这样的语义模块，在原始数据里面混杂着噪音（不同的背景，不同的额外语义信息等），这让分类任务变得困难，如果卷积核不再是固定的nxn，而是具有漂移的能力，可以使卷积是对真正的语义区域做操作，效果应该会更好。 deformable convolutional network[6]在每个卷积层前面加了一个偏置层，可以指定在做卷积时像素点需要偏置的坐标值，我的想法是deformable C3D在训练之后，是否具备映射视频中最关键的语义信息的能力呢，之后会通过实验验证这个想法。







---
### 参考论文：

[1].TS-LSTM and Temporal-Inception: Exploiting Spatiotemporal Dynamics for Activity Recognition.Chih-Yao Ma, Min-Hung Chen et,al. arXiv:1703.10667.https://arxiv.org/abs/1703.10667

[2].Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. Limin Wang, Yuanjun Xiong et al. ECCV 2016.http://wanglimin.github.io/papers/WangXWQLTV_ECCV16.pdf

[3].Deep Local Video Feature for Action Recognition.Zhenzhong Lan et,al. arXiv:1701.07368. https://arxiv.org/abs/1701.07368

[4].Learning Spatiotemporal Features with 3D Convolutional Networks.D. Tran, L. Bourdev et al.ICCV 2015 

[5].Action Recognition with Improved Trajectories.Heng Wang, Cordelia Schmid et al. ICCV 2013

[6].Deformable Convolutional Networks. Jifeng Dai, Haozhi Qi et al. arXiv:1703.06211. https://arxiv.org/abs/1703.06211
