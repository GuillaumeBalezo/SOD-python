def getProposals(I, net, param):
  ##############################################
  # Generate the proposal set for optimization
  ##############################################
  imsz = np.array([I.shape[0], I.shape[1]])
  Ip = np.zeros((param['batchSize'], param['width'], param['height'], 3))
  Ip[0,:,:,:] = prepareImage(I, param)
  scores = net.predict(Ip) ###
  save_val(scores[0], 'cartouche_preds1.mat')
  top_idxs = np.argsort(-scores[0])
  scores = np.take(scores[0], top_idxs)
  BB = param['center'][:, top_idxs]
  P = BB[:, : param['masterImgPropN']].copy()
  S = scores[: param['masterImgPropN']].copy()
  # extract ROIs
  ROI = BB[:, : param['roiN']].copy()
  ROI = postProc(ROI, imsz, param)
  ROI = clusterBoxes(ROI, param) # merge some ROI if needed
  # process ROIs
  Ip = cropImgList_and_Prepare(I.copy(), ROI, param)
  scores = net.predict(Ip)
  save_val(scores.T, 'cartouche_preds2.mat')
  top_idxs = np.argsort(-scores, axis = 1)
  scores = np.take_along_axis(scores, top_idxs, axis = 1)
  for i in range(Ip.shape[0]):
    B = param['center'][:, top_idxs[i, : param['subImgPropN']]]
    roi = ROI[:, i] / np.tile(np.roll(imsz, 1), 2)
    B = getROIBBox(B.copy(), roi)
    P = np.hstack((P, B))
    S = np.hstack((S, scores[i, : param['subImgPropN']]))
  print('getProposals')
  return P, S


def prepareImage(img, param):
  img = preprocess_input(img)
  I = np.expand_dims(cv2.resize(img, (param['width'], param['height']), interpolation = cv2.INTER_LINEAR), axis = 0)
  return I

def clusterBoxes(BB, param):
  if BB.shape[1] < 2:
    ROI = np.copy(BB)
    return ROI

  D = []
  for i in range(BB.shape[1]):
    for j in range(i + 1, BB.shape[1]):
      D.append(1 - getIOUFloat(BB[:, j][:, None].T, BB[:, i])[0])
  Z = cluster.hierarchy.linkage(D)
  T = cluster.hierarchy.fcluster(Z, param['roiClusterCutoff'], criterion = 'distance')
  ROI = np.vstack((BB[:2, T == 1].min(axis = 1, keepdims=True), BB[2:, T == 1].max(axis = 1, keepdims=True))) # initialisation for the for loop
  for i in range(2, T.max() + 1):
    ROI = np.hstack((ROI, np.vstack((BB[:2, T == i].min(axis = 1, keepdims=True), BB[2:, T == i].max(axis = 1, keepdims=True)))))
  print('clusterBoxes')
  return ROI

def postProc(ROI, imsz, param):
  # expand
  w = ROI[2] - ROI[0]
  h = ROI[3] - ROI[1]
  ROI[0] = ROI[0] - 0.5 * w * param['roiExpand']
  ROI[1] = ROI[1] - 0.5 * h * param['roiExpand']
  ROI[2] = ROI[0] + w * (1 + param['roiExpand'])
  ROI[3] = ROI[1] + h * (1 + param['roiExpand'])

  ROI = ROI * np.tile(np.roll(imsz, 1), 2)[:, None]
  ROI[:2] = np.maximum(ROI[:2], 0)
  ROI[2] = np.minimum(ROI[2], imsz[1])
  ROI[3] = np.minimum(ROI[3], imsz[0])


  # removing
  area = (ROI[2] - ROI[0] + 1) * (ROI[3] - ROI[1] + 1)

  ROI = ROI[:, area < (0.9 * imsz[0] * imsz[1])]
  print('postProc')
  return ROI


def cropImgList_and_Prepare(img, roilist, param):
  Ip = []
  if len(roilist.shape) == 1:
    roilist = roilist[:, None]
  for i in range(roilist.shape[1]):
    roi = roilist[:, i]
    img_cropped = img[int(roi[1]) : int(roi[3]) + 1, int(roi[0]) : int(roi[2]) + 1, :]
    img_cropped = preprocess_input(img_cropped)
    Ip.append(cv2.resize(img_cropped, (param['height'], param['width']), interpolation = cv2.INTER_LINEAR)) # ordre des x et y ?
  print('cropImgList_and_Prepare')
  return np.array(Ip)
