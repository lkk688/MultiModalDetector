from utils.plotresults import show_image_bbxyxy
import matplotlib
import os
import time
import cv2

import importlib
from utils import plotresults
importlib.reload(plotresults)

def detectoneimage(imgpath, mydetector):
    start = time.time()
    ori_im = cv2.imread(imgpath)
    imageshape=ori_im.shape
    im_width=imageshape[1]#2720#800
    im_height=imageshape[0]#1530#600
    print("Image width: ", im_width)
    print("Image height: ", im_height)

    im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
    bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)

    end = time.time()
    #show_image_bbxyxy(im, bbox_xyxy, cls_ids, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    #cv2.imwrite('resulttest.jpg', imresult) 
    print("time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))
