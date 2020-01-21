def getIOUFloat(res, gt):
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])
  I = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
  U = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1]) + (gt[2] - gt[0]) * (gt[3] - gt[1])
  iou = I / (U - I)
  return iou

def getROIBBox(B, ROI):
  # Translate the coordinates back into the original image.
  ratio = ROI[2:] - ROI[:2]
  B = B * np.tile(ratio, 2)[:, None]
  B = B + np.tile(ROI[:2], 2)[:, None]
  return B

def expandROI(roi, imsz, margin):
  roi[: 2, :] = roi[: 2, :] - margin
  roi[2 :, :] = roi[2 :, :] + margin
  roi[: 2, :] = np.maximum(roi[: 2, :], 0) ###### 0 or 1 ?
  roi[2, :] = np.minimum(roi[2, :], imsz[1])
  roi[3, :] = np.minimum(roi[3, :], imsz[0])
  return roi

def getIOU(res, gt):
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])

  I = np.maximum((xmax - xmin + 1),0) * np.maximum((ymax - ymin + 1),0)
  U = (res[:, 2] - res[:, 0] + 1) * (res[:, 3] - res[:, 1] + 1) + (gt[2] - gt[0] + 1) @ (gt[3] - gt[1] + 1) ###########
  iou = I / (U - I)
  return iou

def getMaxIncFloat(res, gt):
  #############################################
  # Another way to define window similarity:
  # Intersection over min area.
  #############################################
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])
  I = np.maximum((xmax - xmin), 0) * np.maximum((ymax - ymin), 0)
  U1 = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1])
  U2 = (gt[2] - gt[0]) * (gt[3] - gt[1])
  inc = I / np.minimum(U1, U2)
  return inc

def doNMS(bboxes, thresh):
  if bboxes.size == 0: #not sure
    return bboxes,idx
  tmp = bboxes[:, 0]
  idx = 0
  for i in range(2, bboxes.shape[1]):
    if tmp.shape[0] > 1:
      if getIOUFloat(tmp[:, None].T, bboxes[:, i])[0] < thresh:
        tmp = np.hstack((tmp, bboxes[:, i]))
        idx = np.hstack((idx,i))
    else:
      if np.maximum(getIOUFloat(tmp[:, None].T, bboxes[:, i])) < thresh:
        tmp = np.hstack((tmp, bboxes[:, i]))
        idx = np.hstack((idx,i))

  bboxes = np.copy(tmp)
  return bboxes, idx

def doMMR(bbox, score, lambda_):
  #############################################
  # An implementation of the Maximum Marginal
  # Relevance re-ranking method used in this
  # paper:
      # J. Carreira and C. Sminchisescu. CPMC:
      # Automatic object segmentation using
      # constrained parametric min-cuts. PAMI,
      # 34(7):1312–1328, 2012.
  #############################################
  res = []
  bbox = np.hstack((bbox, score.flatten())) #
  if bbox.size == 0: ###
    return P,S

  res = bbox[0, :]
  bbox[0] = [] ########" array, np.empty ?"
  while bbox.size != 0: ###

    ss = bbox[:, -1]
    for i in range(bbox.shape[0]):
      ss[i] = ss[i] - lambda_ * np.maximum(getIOU(res, bbox[i, :4])) #######
    idx = np.argsort(-ss, axis = 0)
    ss = np.take(ss, idx)
    bbox = bbox[iidx, :]
    res[end() + 1, :] = np.hstack((bbox[0, :4], ss[0])) #######
    bbox[0] = [] #########" array"


  P = res[:, :4].T ########
  S = res[:, -1]
  return P, S

def initModel(param):
  '''
  Input:
      weights_path: path of h5 file with the weights
      center_path: path of npy file with the center bounding boxes
  Output:
      model: SOD cnn model
  '''
  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
  flat = tf.keras.layers.Flatten()(vgg16.output)
  fc6 = tf.keras.layers.Dense(4096, activation='relu')(flat)
  fc7 = tf.keras.layers.Dense(4096, activation='relu')(fc6)
  fc8 = tf.keras.layers.Dense(100, activation='sigmoid')(fc7)
  model = tf.keras.models.Model(inputs=[vgg16.input], outputs=[fc8])
  model.load_weights(param['weightsFile'])
  model.trainable = False
  return model
