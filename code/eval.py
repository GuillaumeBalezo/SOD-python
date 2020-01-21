############################ Evaluation
def calcAP(rec, prec, interval=None):
  ##############################################
  # Calculating average precision using 11 point
  # averaging.

  # We first linearly interpolate the PR curves,
  # which turns out to be more robust to the
  # number of points sampled on the PR cureve.
  ##############################################
  if not interval:
    interval = np.arange(0,1,0.1)

  # linear interpolation
  ii = np.argsort(rec) # axis ?
  rec = np.take(rec, ii)
  prec = np.take(prec, ii)
  rec, ii = np.unique(rec, return_index = True)
  prec = np.take(prec, ii)
  Rq = np.arange(0, 1, 0.01)
  Pq = np.interp(rec, prec, Rq) ####
  Pq = np.nan_to_num(Pq, nan = 0)
  prec = np.copy(Pq)
  rec = np.copy(Rq)
  ap = 0
  for t in interval:
    p = np.maximum(prec[rec >= t])
    if p.size == 0:
      p = 0
    ap = ap + p / len(interval)
  return ap

def evaluateBBox(imgList, res): ########################### deak with MSO dataset (ex: .anno)
  ##############################################
  # Function for evaluation.

  # It outputs the number of hits, prediction
  # and ground truths for each image.
  ##############################################
  TTP = np.zeros((len(res), len(imgList)))
  NPred = np.zeros((len(res), len(imgList)))
  NGT = np.zeros((1, len(imgList)))
  for i in range(len(res)):
    pred_num = np.zeros((1, len(imgList)))
    TP = np.zeros((1, len(imgList)))
    for j in range(1, len(imgList)):
      NGT[j] = size(imgList[j].anno,1)
      pred_num[j] = res[i][j].shape[1]
      bboxes = getGTHitBoxes(res[i][j], imgList[j].anno, 0.5)
      TP[j] = bboxes.shape[1]
    TTP[i, :] = TP
    NPred[i, :] = pred_num

  TP = np.copy(TTP)
  return TP, NPred, NGT

def getGTHitBoxes(bboxes, anno, thresh):
  ##############################################
  # This function returns the correct detection
  # windows given the ground truth.
  ##############################################
  res = []
  if anno.size == 0 or bboxes.size ==0:
    return res

  for i in range(anno.shape[0]):
    iou = getIOU(bboxes.T,anno[i, :])
    idx = iou.argmax()
    score = iou[idx]
    if score > thresh:
      res = np.concatenate((res,bboxes[:,idx]))
      bboxes[:, idx] = [] ###### array !!
  return res
