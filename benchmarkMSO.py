##############################################
# Benchmark methods on the MSO dataset
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
##############################################

param = getParam('VGG16')
net = initModel(param)

load dataset/MSO/imgIdx

cacheDir = 'propCache'
if not exist(cacheDir, 'dir'):
  mkdir(cacheDir)

props = []
flag = False
cacheName = ['MSO_' + param['modelName'] + '.mat'] #

# load precomputed proposals if possible
if exist(fullfile(cacheDir,cacheName), 'file'):
  print('using precomputed proposals.\n')
  load(fullfile(cacheDir, cacheName)) #
  flag = True


legs = []

## evaluate the MAP method
print('run the MAP method')
lambda_ = [0,0.000001,0.0001,0.01:0.01:0.1,0.1:0.1:1]
res = []
for j in range(len(imgIdx)):
  I = imreadRGB(fullfile('dataset/MSO/img/',imgIdx(j).name)) ##
  imsz = [I.shape[0], I.shape[1]]

  # load precomputed proposals
  if flag:
    P = props(j).P
    S = props(j).S
  else:
    [P, S] = getProposals(I, net, param)
    props(j).P = P
    props(j).S = S

  if j % 100 == 0:
    print('processed {} images'.format(j))


  for i in range(len(lambda_)):
    param['lambda'] = lambda_[i]
    param['gamma'] = 10 * lambda_[i]
    tmpRes = propOpt(P, S, param)

    # scale bboxes to full size
    tmpRes = tmpRes * np.tile(imsz.roll(1), 2).T)
    res{i}{j} = tmpRes # !


K.clear_session()

if not flag:
  save(fullfile(cacheDir, cacheName), 'props', '-v7.3') # !

figure # !
hold on # !

TP, NPred, NGT = evaluateBBox(imgIdx, res)

P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(0), 0.01)
plot(R,P,'r') # !
ap = calcAP(R, P)
legs.append('MAP: {}'.format(ap))

## evaluate the NMS baseline
print('run the NMS baseline')
thresh = np.arange(0, 1, 0.02) # !
res = []
for j in range(len(imgIdx)):
  I = imreadRGB(fullfile('dataset/MSO/img/',imgIdx(j).name))
  imsz = [size(I,1), size(I,2)]
  P = props(j).P
  S = props(j).S
  if j % 100 == 0:
    print('processed %d images\n',j)

  # scale bboxes to full size
  P = P * np.tile(imsz.roll(1), 2).T
  idx = np.argsort(-S)
  S = np.take(S, idx)
  P = np.take(P, idx, axis = 1)
  P, sidx = doNMS(P, 0.4)
  S = S[sidx]

  for i in range(len(thresh)):
    tmpRes = P(:, S>=thresh(i))
    res{i}{j} = tmpRes

TP, NPred, NGT = evaluateBBox(imgIdx, res)
P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(), 0.01)
plot(R,P,'b') # !
ap = calcAP(R,P)
legs.append('NMS: {}'.format(ap))

## evaluate the MMR baseline
print('run the MMR  baseline')
thresh = -1.0:0.01:1.0 #!
res = []
for j in range(len(imgIdx)):
  I = imreadRGB(fullfile('dataset/MSO/img/',imgIdx(j).name))
  imsz = [I.shape[0], I.shape[1]]
  P = props(j).P #!
  S = props(j).S #!
  if j % 100 == 0:
    print('processed {} images'.format(j))


  # scale bboxes to full size
  P = bsxfun(@times, P, imsz([2 1 2 1]).T)
  [S,idx] = sort(S, 'descend')
  P = P(:,idx)
  [P, S] = doMMR(P.T, S, 1.0)
  for i in range(len(thresh)):
    tmpRes = P[:, S > thresh[i]] #!
    res{i}{j} = tmpRes #!

TP, NPred, NGT = evaluateBBox(imgIdx, res)
P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(), 0.01)
plot(R,P,'g')#!
ap = calcAP(R,P)
legs.append('MMR: {}'.format(ap))

grid on #!
legend(legs) #!
title('PR Curves on the MSO Dataset ({})'.format(param.modelName)) #!
