import argparse
import glob
import os
import time
import cv2

from MyDetector import TF2Detector
from utils import detectimage

class detectorargs:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    modelbasefolder = './models/ModelZoo/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model/'
    modelfilename='faster_rcnn_resnet50_v1_640x640_coco17_tpu-8' #not used
    #showfig='True'
    labelmappath = './models/research/object_detection/data/mscoco_label_map.pbtxt'
    threshold = 0.3

if __name__ == "__main__":
    mydetector = TF2Detector.MyTF2Detector(detectorargs)

    imgpath=os.path.join('.', "test.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)
