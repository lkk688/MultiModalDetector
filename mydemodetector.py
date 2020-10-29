import argparse
import glob
import os
import time
import cv2

from MyDetector import TF2Detector
from utils import detectimage

class detectorargs:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    modelbasefolder = '../models/ModelZoo/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model/'
    modelfilename='faster_rcnn_resnet50_v1_640x640_coco17_tpu-8' #not used
    #showfig='True'
    labelmappath = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
    threshold = 0.3

if __name__ == "__main__":
    mydetector = TF2Detector.MyTF2Detector(detectorargs)

    #imgpath=os.path.join('./testdata', "traffic1.jpg")
    imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "11901761444769610243_556_000_576_000_1515475579357063_FRONT.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

    outputvideopath='videoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectimage.detectimagefolder_tovideo(folderpath, mydetector, outputvideopath)
