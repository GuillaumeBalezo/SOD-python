def propOpt(bboxes, bboxscore, param):
  ##################################################
  # The main function for the subset optimization
  ##################################################


  # for the special case when lambda == 0
  if param['lambda'] == 0:
    stat = {}
    res = bboxes.copy()
    stat['O'] = np.arange(bboxes.shape[1]).reshape(-1, 1)
    return res, stat

  stat = doMAPForward(bboxes, bboxscore.astype(float), param)
  if stat['O'].size > 1:
    stat = doMAPBackward(bboxes, bboxscore.astype(float), param, stat)

  # We use the second output to intialized the optimization again
  if param['perturb'] and len(stat['O']) > 1:
    # use the second output to initialize the forward pass
    statTmp = doMAPEval(bboxes, bboxscore.astype(float), param, stat['O'][1], stat['W'], stat['BGp'])
    statTmp = doMAPForward(bboxes, bboxscore.astype(float), param, statTmp)
    if statTmp['f'] > stat['f']:
      stat = statTmp.copy()
  res = np.take(bboxes, stat['O'].flatten(), axis = 1).copy()
  return res, stat


def doMAPForward(B, S, param, stat = None):
  if B.size == 0:
    print('Empty proposal set')
    stat = {}
    return stat

  nB = B.shape[1]
  if not stat:
    # initialization
    stat = {}
    stat['W'] = np.array([])
    stat['Xp'] = np.array([])
    stat['X'] = np.zeros((nB, 1))
    # construct W
    stat['W'], stat['Xp'] = getW(B, S, param)
    stat['BGp'] = stat['Xp'].copy()
    stat['nms'] = np.zeros((B.shape[1], 1))
    stat['f'] = stat['Xp'].sum()
    stat['O'] = np.array([], dtype=int)

    ## loop
    while len(stat['O']) < min(param['maxnum'], nB):
      V = np.maximum(stat['W'] - stat['Xp'][:, None], 0)
      #V = np.maximum(stat['W'] - np.tile(stat['Xp'], (1,nB)), 0)
      scores = V.sum(axis = 0) + stat['nms'].flatten().T
      vote = np.argmax(scores)
      score = scores[vote]
      if score == 0:
        break
      tmpf = stat['f'] + score + param['phi']

      if (tmpf > stat['f']):
        mask = V[:, vote] > 0
        stat['X'][mask] = vote
        stat['O'] = np.append(stat['O'], vote)[:, None]
        stat['Xp'][mask] = stat['W'][mask, vote]
        stat['f'] = tmpf
        stat['nms'] = stat['nms'] + param['gamma'] * getNMSPenalty(B, B[:, vote])[:, None]
      else:
        break
  return stat

def doMAPBackward(B, S, param, stat):
  while stat['O'].size != 0:
    flag = False
    bestStat = stat.copy()
    for i in range(len(stat['O'])):
      O = stat['O'].copy()
      O = np.delete(O, i)
      statTmp = doMAPEval(B, S, param, O, stat['W'], stat['BGp'])
      if statTmp['f'] > bestStat['f']:
        flag = True
        bestStat = statTmp.copy()
    stat = bestStat.copy()
    if not flag:
      break

  return stat

def doMAPEval(B, S, param, O, W = None, BGp = None):
  ##############################################
  # This function evaluate the target function
  # given a output window set.
  ##############################################
  statTmp = {}
  statTmp['W'] = np.array([])
  statTmp['Xp'] = np.array([])

  statTmp['X'] = np.zeros((B.shape[1], 1))
  if type(W).__name__ == 'NoneType' or type(BGp).__name__ == 'NoneType': ###better
    # construct W
    statTmp['W'], statTmp['BGp'] = getW(B, S, param)
  else:
    statTmp['W'] = W.copy()
    statTmp['BGp'] = BGp.copy()

  statTmp['nms'] = np.zeros((B.shape[1],1))
  statTmp['O'] = O.copy()
  statTmp['f'] = O.size * param['phi']
  for i in range(O.size):
    statTmp['f'] = statTmp['f'] + statTmp['nms'][O[i], 0]
    statTmp['nms'] = statTmp['nms'] + param['gamma'] * getNMSPenalty(B, B[:, O[i]])[:, None]

  if O.size == 0:
    statTmp['f'] = statTmp['f'] + np.sum(BGp)
    return statTmp

  tmp_val = statTmp['W'][:, O]
  ass = tmp_val.argmax(axis = 1)
  Xp = tmp_val[ass]
  #Xp = np.take(tmp_val, ass, axis = 1)
  print(Xp.shape)
  print(BGp.shape)
  mask = Xp > BGp.reshape(-1, 1)
  statTmp['X'][mask] = ass[:, None][mask]
  statTmp['Xp'] = np.maximum(Xp, BGp)
  statTmp['f'] = statTmp['f'] + statTmp['Xp'].sum()
  return statTmp

def getNMSPenalty(B, b):
  p = -0.5 * (getMaxIncFloat(B.T, b) + getIOUFloat(B.T, b))
  return p

def getW(B, S, param):
  #######################################################
  # Precompute all w_{ij}
  # Xp is the likelihood of the optimal assignments given
  # the current output set
  #######################################################
  P = np.zeros((B.shape[1], B.shape[1]))
  for i in range(B.shape[1]):
    P[i, :] = getIOUFloat(B.T, B[:, i].T)
  P = P * S[:, None]
  P = np.hstack((P, param['lambda'] * np.ones((B.shape[1], 1))))
  np.seterr(divide='ignore')
  P = P / P.sum(axis = 1, keepdims=True)
  W = np.log(P)
  Xp = W[:, -1]
  W = W[:, : -1]
  return W, Xp

def refineWin(I, res, net, param):
  ########################################################
  # A heuristic way to further refine the output windows

  # For each small output window, we run our method on the
  # this ROI again and extract the output that has the
  # largest IOU with the orignal window for replacement.

  # NMS is further applied to remove duplicate windows.
  ########################################################
  imsz = np.hstack([I.shape[0], I.shape[1]])
  param['lambda'] = 0.05
  for i in range(res.shape[1]):
    bb = res[:, i]
    bbArea = (bb[2] - bb[0]) * (bb[3] - bb[1])
    if bbArea < (0.125 * imsz[0] * imsz[1]):
      margin = (bb[2] - bb[0] + bb[3] - bb[1]) * 0.2
      bb = expandROI(bb, imsz, margin).astype(int)
      Itmp = I[bb[1] : bb[3],bb[0] : bb[2], :] # ?
      Ptmp, Stmp = getProposals(Itmp, net, param)
      restmp = propOpt(Ptmp, Stmp, param)
      if not isempty(restmp): ##########
        restmp = getROIBBox(restmp, bb)
        __, ii = np.max(getIOUFloat(restmp.T, res[:, i]),nargout=2) #####" argmax"
        res[:, i] = restmp[:, ii]

  res = doNMS(res, 0.5)
  return res
