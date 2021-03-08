__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy

""" 3 steps:
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
"""

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
                    # 第一个整体范围， 小目标(small),        中目标(middle)       大目标(large)
    #  p.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        # 这4个属性在每次_prepare 又重新初始了
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        # annotations {(image_id, category_id):[anno,..]}
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation

        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        # len self.states=12 after summarize
         """ self.stats
            AP(default):iou=[0.5,1), recall=[0,1), num_class=80, area='all', maxdet=100
            0:AP(all)
            1:AP(iou>=0.5)
            2:AP(iou>=0.75)
            3:AP(area='small')
            4:AP(area='medium')
            5:AP(area='large')
            Recall(default):iou=[0.5,1),num_class=80, area='all', maxdet=100
            6:AR(maxdet=1)
            7:AR(maxdet=10)
            8:AR(maxdet=100)
            9:AR(area='small')
            10:AR(area='medium')
            11:AR(area='large')

        """
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        # actually, self.evalImgs is reset to a list[dict] as follows in self.evalute()
        """
        {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                # 前3个作为key
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm, # [T, D]
                'gtMatches':    gtm, # [T, G]
                'dtScores':     [d['score'] for d in dt], # [D]
                'gtIgnore':     gtIg, # [G]
                'dtIgnore':     dtIg, # [D]
            }
        """

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        # default:True, 使用类别标签
        if p.useCats:
            # p.imageIds 是所有图片的id，p.catIds 所有类别的id

            # 这样过滤岂不是跟加载全部一样 ？？？
            # 除非有些图片是没有任何类别的，就会被过滤掉。
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag: 用来干啥呢？
        for gt in gts:
            # 如果没有'ignore' 属性，就设置为0
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            # 以 元组(image_id, category_id) 作为key，
            # 实际上每个(#image_id, category_id) 这样的pair有限吧，不是每个图片都有所有类别
            # 假如有类别1,2,3,4,5
            # 对于图片001，只有1,1，2 两个类别共3个对象，那么对于key(001,3) 应该要报错
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        """
        self.eval = {
            'params': p,# 保存参数配置
            'counts': [T, R, K, A, M], # number of threshold, recThres, class, areaRng, maxDets
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        T           = len(p.iouThrs) # 10
        R           = len(p.recThrs) # 100 recall [0., 1, 0.01]
        K           = len(p.catIds) if p.useCats else 1 # 类别数 80
        A           = len(p.areaRng) # 面积种类 4
        M           = len(p.maxDets) # M [1,10,100]
        # precision[t,:,a,m] 保存的是在阈值索引为t,面积范围索引为a,单张最大检测数索引m时
        # 不同召回率阈值 self.recThres:[101] 对应的precision值[101]
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        # 对应于上面的precision的score,注意score是该精度的下限
        scores      = -np.ones((T,R,K,A,M))
        recall      = -np.ones((T,K,A,M))
        """

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            # self.computeIoU is a function
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        # 计算iou
        # 遍历每个图片，每个图片遍历每个类别
        # return: iou:[m,n] where m: num of dt, n: num of gt
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}
        """
        (imgId, catId): []
        or [m,n]
        []: 如果图片imgId 没有catId, 或者预测中没有catId 都返回一个空列表
        [m,n]:对 m,n 的理解
        mxn,保存每个dt 的box 和 每个 gt 的box的iou, 因为其实不知道每个dt box 
        对应哪个gt box, 但是他们都是同一个类别，其实只要能够就近计算就行，只有和每个
        gt box 计算iou, 才能够知道每个dt box 更靠近哪个，然后再为每个dt box 选一个gt box
        """
       
        # self.evaluateImg is a function
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
                        # 第一个整体范围(all)，  小目标(small),        中目标(middle)       大目标(large)
        #  p.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        #  areaRng    - [...] A=4 object area ranges for evaluation
        #  评估每个类别的不同size，遍历所有图片查找
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        # shape of self.evalImgs:[num_catId*num_areaRng*num_imgId]
        # evaluateImg a dict
        """
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                # 前3个作为key
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm, # [T, D]: 匹配情况
                'gtMatches':    gtm, # [T, G]: 与dtMatches 对应，在accumulate 没有用到。
                'dtScores':     [d['score'] for d in dt], # [D]
                'gtIgnore':     gtIg, # [G]
                'dtIgnore':     dtIg, # [D]
            }
        """
        self._paramsEval = copy.deepcopy(self.params) # 复制来干嘛？？？
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
    # 处理一个图片的其中一个类别
    # self._gts[image_id, cat_id]:list[anno] 存储的是对于图片image_id, 关于类别cat_id的 gt anno
    # self._dts 对应于检测的结果
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            # 实际上self._gts 的key(#image_id, category_id) 
            # 这样的pair有限吧，不是每个图片都有所有类别
            # 这里的全组合(imgId, catId) 不会有key error 吗
            # 答：特别地，对于defaultdict(list), 对于不存在的key,不报错，统一返回[]
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        # 如果都为为空，返回空列表 
        # 就是都不存在，不用处理
        if len(gt) == 0 and len(dt) ==0:
            return []
        # 在后面(iou in _mask.pyx)，for bbox, 如果有其中一个为空，其实也是返回[],

        # argsort 从小到大排序,通过负号，实现了从大到小排序
        # 万一是 gt!=[], dt=[], 那这个-d['score'] 不就key_error, 不用非空检验一下吗
        # 答：# 如果为空，根本无法进入循环，所以也不会执行下面的话
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        # for onenet: dt<=100(p.maxDets[-1])
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        # 获取bbox
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        """
        单个的对象（iscrowd=0)可能需要多个polygon来表示，比如这个对象在图像中被挡住了。
        而iscrowd=1时（将标注一组对象，比如一群人）的segmentation使用的就是RLE格式。
        """
        iscrowd = [int(o['iscrowd']) for o in gt]
        # m = len(d), n = len(g)
        # return iou:[m,n]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious
    """
    aRng: 给定gt 的area 范围
    """
    # 2021.3.3
    # 指定类别指定大小范围aRng的图片imgId 的评估
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # 用g['_ignore'] 来标记不符合面积要求的g
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # 标记为1的放在后面
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        # 对detection anno 排序
        # 为了保持跟计算iou的一致
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        # self.ious[imgId, catId]: [] or [dm, gn]: dn = len(dt), gn = len(gt)
        # 根据gtind 调整 gt 对应的位置, 因为gt 位置变了
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        #p.iouThrs [0.5,0.55,...0.95]
        T = len(p.iouThrs) # 10
        G = len(gt)
        D = len(dt)
        # 如果(T,G) 有任意一个为0，创建的的是空列表
        gtm  = np.zeros((T,G)) # gtm[i,j] 保存对于每个阈值i，配对dt box 的id： j
        # dtm: 也可以理解为保存了是否与gt 匹配成功(pos)，默认是没有匹配成功(neg)。
        dtm  = np.zeros((T,D)) # dtm[i,j],保存对于每个阈值i, 配对gt box 的id：j, 0 为没有匹对的
        gtIg = np.array([g['_ignore'] for g in gt]) # 记录gtIg 一些不符合条件的gt, 在匹配过程可忽视
        dtIg = np.zeros((T,D)) # dtIg[i,j] 对应与dtm,标志配对的(j, dtm[i,j]) 是符合要求，=1 表示不符合
        # not len(ious==0): 只有gt,dt 都不为空才执行
        if not len(ious)==0:# 
            """
            个人觉得缺少了else:return None (for dt or gt = [])
            答： 特别地，当dt=0, 那么dtm=[],也就是缺少检测，在accumulate的时候，concat dtm, 不影响，只是recall肯定会低
            当dt=/0, gt=0, 这个就是多检测了，dtm=[T,D], 全都是没有配对的，在accumulate 的时候，concat dtm 没问题，知识precision 肯定会低
            """
            # 枚举threshold
            for tind, t in enumerate(p.iouThrs):
                # 枚举dt
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1 # 保存匹配的gt的index
                    # 枚举gt
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        # 已经匹配过了，not a crowd: 不是多个对象，所以就可以跳过了
                        #  注意dt 是按分数从高到低排序的，自然有优先选择权
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # 如果dt已经匹配到了gt(m>-1), 现在已经遍历到该忽视gt(gtIg[gind]==1),就停止
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        # 要找到比当前iou更好的才执行更新
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind] # 重新更新当前最好的iou
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m] # 记录匹配上的gt 对应的_ignore标志，如果为0，就是符合要求
                    dtm[tind,dind]  = gt[m]['id'] # 记录dt匹配上的gt的id号
                    gtm[tind,m]     = d['id'] # 记录gt 匹配上dt 的id号

        # here
        # set unmatched detections outside of area range to ignore
        # 标记不在范围内的dt bbox
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        # dtm==0,没匹配的为True,匹配的为False (找出没匹配的)
        # np.repeat(a,repeats, axis), axis=0, 增加的是行，axis=1增加的是列
        # 显然这里是为了得到T 行 a
        # logical_and: 只有没匹配的，面积又不在范围内的才为 True，在范围的为False,0
        # logical_or(dtIg,[n,a]) orginal: dtIg: 符合要求的为False, 不符合的为True
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # 这样一处理，为True的有(匹配的但gt被ignored的，没匹配的&dt面积不在范围的)
        # 所以其实是想把符合面积范围的保留？吧
        # 为什么不一开始就去掉不符合面积要求的dt...

        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm, # [T, D]
                'gtMatches':    gtm, # [T, G]
                'dtScores':     [d['score'] for d in dt], # [D]
                'gtIgnore':     gtIg, # [G]
                'dtIgnore':     dtIg, # [D]
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs) # 10
        R           = len(p.recThrs) # 100 recall [0., 1, 0.01]
        K           = len(p.catIds) if p.useCats else 1 # 类别数 80
        A           = len(p.areaRng) # 面积种类 4
        M           = len(p.maxDets) # M [1,10,100]
        # precision[t,:,a,m] 保存的是在阈值索引为t,面积范围索引为a,单张最大检测数索引m时
        # 不同召回率阈值 self.recThres:[101] 对应的precision值[101]
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        # 对应于上面的precision的score,注意score是该精度的下限
        scores      = -np.ones((T,R,K,A,M))
        recall      = -np.ones((T,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        # 集合化
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        # 为什么不是存k, _pe.catIds 和 p.catIds本来就是同一个东西，当然全都在
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        # 除了maxDets 存储值，其他都是索引
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        # 转化为索引列表，主要是方便计算在特定的k0,a0,前面已经处理过多少 item of self.evalImgs
        # 此外precision, scores, recall 都是通过他们的索引去访问的，而不是实际元素

        I0 = len(_pe.imgIds) # 图片总数量
        A0 = len(_pe.areaRng) # 面积类型数量
        # self.evalImgs:[num_catId*num_areaRng*num_imgId]
        # :[K0,A0,I0]
        # retrieve E at each category, area range, and max number of detections
        # k_list 本来就是对应的索引，所以k,k0 不就是同一个东西吗
        # 遍历类别索引
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0 # 处理新类别索引为k0 的时候，已经处理了多少 item of self.evalImgs
            # 遍历面积类型索引
            for a, a0 in enumerate(a_list):
                Na = a0*I0 # 从处理类别索引为k0开始,面积索引为a0的时候，已经处理了多少 item of self.evalImgs
                # 遍历检测数[1,10,100]
                for m, maxDet in enumerate(m_list):
                    # 显然这个结果要重复取3次
                    # 获取当 catid = catId[k0] and area = areaRng[a0] 的时候，所有图片的处理结果
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    # 设valid_num = len(E)
                    if len(E) == 0:
                        continue
                    # 根据maxDet, 把所有图片的前maxDet 的分数合并
                    # [valid_dt]
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    # 在evaluateImg() 的时候就已经排过序了，为什么又排？
                    # 因为这里是合并了所有图片的结果
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    #                      # [T,<maxDet] + [T, <maxDet] + ..
                    # [T, valid_dt] # valid_dt, 所有图片中，有效的检测数
                    # 注意这里是按分数排序，可能有些dtm[i]=0(没有匹配到gt的)
                    # 特别地，所有被选中的dt, 也就是dtm, 都认为是预测的正样本
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    # 获取对应是否要忽视的标志
                    # [T, valid_dt]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    # [T, valid_gt]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    # 计算gtIg 为0的个数，也就是有效 gt 个数
                    npig = np.count_nonzero(gtIg==0 )
                    # 如果没有gt, 但dtm 不为0，那么tp 必然为0，presion,recall 比如为0，这种在后面求mean会拖低平均值
                    # 但这样跳过，是不是就取默认值为-1了，后在求mean的时候似乎有点不妥。
                    # 说白了就是 gt=[], dt！=[] 的情况没处理好。按照代码的思路，直接ignore了。
                    """
                    感觉应该这样：
                    if npig == 0 and len(dtm=0):
                        contiue
                    """
                    if npig == 0:
                        continue
                    # binary: 标记符合要求的检测(1)和不符合要求的检测(0)
                    # 只有dtIg=0 的位置对应的非0才是正样本，对应的0为负样本。那写dtIg=1对应的那些
                    # box在这里并不影响计算结果。因为 pr 的分母是fp+tp,而不是box数
                    # 真正样本, 
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    # 假正样本，那些dtm[i]=0, 就是那个box 预测错了，所以是假正样本
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    # 累计求和
                    # [T, valid_dt]
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    # 遍历每个阈值的情况
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        #[valid_dt]
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp) # valid_dt, 其实放在外面比较好，因为是重复的
                        rc = tp / npig # 召回率 tp / 总的正样本数
                        pr = tp / (fp+tp+np.spacing(1)) # 精确率 np.spacing(1) 产生一个很小的正数
                        q  = np.zeros((R,)) # 用来存在每个Recall断点(p.recThrs)对应的precision
                        ss = np.zeros((R,)) # ss(ScoreSorted):记录q对应的score

                        if nd:
                            # 最后一个
                            recall[t,k,a,m] = rc[-1]
                        else:
                            # 如果dtm=[], 但是gt=/[],就会出现这种情况
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()
                        # smooth precision, 因为实际上pr是一个多段起伏的趋势
                        # e.g. [1.,1.,0.6,0.7,0.8,0.3,0.4,0.5,0.1,0.2,...]
                        # 平滑方法，每个位置取其右边比自己大的数中最大的数代替自己
                        # smooth -> [1.,1.,0.8,0.8,0.8,0.5,0.5,0.5,0.2,0.2]
                        # 这样就是一个整体下降的趋势
                        # 按照下面的算法就可以达到上面的要求
                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]
                        # np.searchsorted, 给定一个升序的数组rc, 给定一个列表，返回插入列表中每个元素
                        # 应该位于的位置，但是并没有真正插入
                        # 感觉效果跟bisect一样。
                        # 相当于找出特定(召回率)的分段点。
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                # q[ri] 表示对应与p.recThrs[ri] 召回率的 precision
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi] # ss 保存对应的分数
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))
    # 2021.3.4
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            #for ap :"Average Precision" (AP) @[IoU=[0.5:0.95] | area=all | maxDets=100] = 0.9,
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            # 如果iou不指定，那就是[0.5:0.95]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            # 寻找面积范围，最大检测数的索引 idx
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU,
                if iouThr is not None:
                    # 如果不为空，就选择对应iou阈值的索引t
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t] # s:[R,K,A,M]
                s = s[:,:,:,aind,mind]
                # 个人觉得此处应该改成这样：
                """
                if iouThr is not None:
                    t = np.whwere(iouThr == p.iouThrs)[0]
                    s = s[t,:,:,aind,mind]
                else:
                    s = s[:,:,:,aind,mind]
                """
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t] # s[K,A,M]
                s = s[:,:,aind,mind]
                # 同上，个人觉得应该如下
                """
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t,:,aind,mind]
                else:
                    s = s[:,:,aind,mind]
                """
            if len(s[s>-1])==0:
                # s>-1: return a bool array with shape as s
                mean_s = -1
            else:
                # 用 s>-1:作为索引，就是提取满足要求的元素组成一个一维列表
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            # iou>=0.5，所有召回率(recall), 所有类别，所有面积尺寸(all), 最大检测数为100 的平均AP
            stats[0] = _summarize(1)
            # iou>=0.5, 所有召回率(recall), 所有类别，所有面积尺寸(all), 最大检测数为100 的平均AP
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # iou=0.75 所有召回率(recall), 所有类别，所有面积尺寸(all), 最大检测数为100 的平均AP
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # 面积尺寸为small(0,32**2), iou>=0.5, 所有召回率(recall), 所有类别 最大检测数为100 的平均AP
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # 面积尺寸为medium(32**2, 96**2), ....
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # 面积尺寸为large(96**2, 1e5**2), ....
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # iou>=0.5, 最大检测数为1, 所有类别，所有面积尺寸 的平均召回率
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # iou>=0.5, 最大检测数为10，....
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # iou>=0.5, 最大检测数为100，....
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # iou>=0.5, 最大检测数为100，面积尺寸为small,所有类别，的平均召回率
            # 下面同理
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
