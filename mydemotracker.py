import argparse
import glob
import os
import time
import cv2

from MyDetector import TF2Detector
from MyDetector import Detectron2Detector
from MyDetector import TorchVisionDetector
from utils import detectandtrack, detectimage

from MyTracker.deep_sort import DeepSort

class Detectron2detectorargs:
    modelname = 'faster_rcnn_X_101_32x8d_FPN_3x'
    modelbasefolder = '../ModelOutput'
    modelfilename='waymo_fasterrcnnx101_detectron2model_final.pth' #
    showfig='True'
    FULL_LABEL_CLASSES=['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    threshold = 0.3

class deepsortargs:
    modelname = 'deepsort'#not used here
    modelbasefolder = '../models/ModelZoo/deepsortcheckpoint'
    modelfilename='original_ckpt.t7' #ckpt.t7

def testDetectron2Detector(detectorargs):
    mydetector = Detectron2Detector.MyDetectron2Detector(detectorargs)
    imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "11901761444769610243_556_000_576_000_1515475579357063_FRONT.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

    #Test a folder of image and output a video
    outputvideopath='detectron2videoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectimage.detectimagefolder_tovideo(folderpath, mydetector, outputvideopath)

if __name__ == "__main__":
    deepsort_checkpoint='/home/kaikai/Documents/MyDetector/ModelOutput/deepsortcheckpoint/ckpt.t7'
    deepsort = DeepSort(deepsort_checkpoint, use_cuda=True)

    mydetector = Detectron2Detector.MyDetectron2Detector(Detectron2detectorargs)

    outputvideopath='detectron2trackvideoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectandtrack.trackimagefolder_tovideo(folderpath, mydetector, deepsort, outputvideopath)
