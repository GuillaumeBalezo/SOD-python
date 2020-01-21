# default: using GoogleNet
# other option: VGG16 which is used in the paper
param=getParam('VGG16')
# param = getParam('VGG16');
net=initModel(param)
