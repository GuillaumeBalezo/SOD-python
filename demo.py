import numpy as np
from scipy import cluster
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
from PIL import Image
import cv2
import time
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img


fn = 'birds.jpg'
I = np.array(load_img(fn))
print(I.shape)

imsz = [I.shape[0], I.shape[1]]
param = getParam(modelName = 'SOD_python', weights_path = '/content/drive/My Drive/Meero/weights/sod_cnn_weights.h5', center_path = '/content/drive/My Drive/Meero/layout_aware_subnet/SOD_python/center100.npy')

K.clear_session()
net = initModel(param)

tic = time.time()
P, S = getProposals(I, net, param)
res, _ = propOpt(P, S, param)
# scale bboxes to full size
res = res * np.tile(np.roll(imsz, 1), 2)[:, None]
# optional window refining process
toc = time.time()
print('Time elapsed: {}'.format(round(toc - tic, 2)))
for i in range(res.shape[1]):
  rect = res[:, i]
  #rect[2:] = rect[2:] - rect[:2] + 1
  rect = rect.astype(int)
  I = cv2.rectangle(I, (rect[0], rect[1]), (rect[2], rect[3]), color = (255,0,0), thickness = 5)

cv2_imshow(I)

# optional window refining process
resRefine = refineWin(I, res, net, param)
I = np.array(load_img(fn))
for i in range(len(resRefine)):
  rect = resRefine[:, i]
  rect[2:] = rect[2:] - rect[:2] + 1
  rect = rect.astype(int)
  I = cv2.rectangle(I, (rect[0], rect[1]), (rect[2], rect[3]), color = (255,0,0))

cv2_imshow(I)
