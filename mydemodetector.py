import argparse
import glob
import os
import time
import cv2

from MyDetector import TF2Detector
from MyDetector import Detectron2Detector
from MyDetector import TorchVisionDetector
from utils import detectimage

class TF2detectorargs:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    modelbasefolder = '../models/ModelZoo/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model/'
    modelfilename='faster_rcnn_resnet50_v1_640x640_coco17_tpu-8' #not used
    #showfig='True'
    labelmappath = '../models/research/object_detection/data/mscoco_label_map.pbtxt'
    threshold = 0.3

class Detectron2detectorargs:
    modelname = 'faster_rcnn_X_101_32x8d_FPN_3x'
    modelbasefolder = '../ModelOutput'
    modelfilename='waymo_fasterrcnnx101_detectron2model_final.pth' #
    showfig='True'
    FULL_LABEL_CLASSES=['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    threshold = 0.3

class TorchVisiondetectorargs:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    modelbasefolder = '../ModelOutput'
    modelfilename='waymo_fasterrcnn_resnet50_fpnmodel_27.pth'
    showfig='True'
    FULL_LABEL_CLASSES = [
    'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
    ]
    threshold = 0.3

def testTorchVisionDetector(detectorargs):
    mydetector = TorchVisionDetector.TorchVisionFasterRCNNDetector(detectorargs)
    imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "11901761444769610243_556_000_576_000_1515475579357063_FRONT.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

    #Test a folder of image and output a video
    outputvideopath='torchvisionvideoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectimage.detectimagefolder_tovideo(folderpath, mydetector, outputvideopath)
    

def testDetectron2Detector(detectorargs):
    mydetector = Detectron2Detector.MyDetectron2Detector(detectorargs)
    #imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "11901761444769610243_556_000_576_000_1515475579357063_FRONT.jpg")
    imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "10247954040621004675_2180_000_2200_000_1553810174060909_FRONT.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

    #Test a folder of image and output a video
    outputvideopath='detectron2videoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectimage.detectimagefolder_tovideo(folderpath, mydetector, outputvideopath)
    
def testTF2Detector(detectorargs):
    mydetector = TF2Detector.MyTF2Detector(detectorargs)

    #Test a single image
    #imgpath=os.path.join('./testdata', "traffic1.jpg")
    imgpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/validation_0000', "11901761444769610243_556_000_576_000_1515475579357063_FRONT.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

    #Test a folder of image and output a video
    outputvideopath='TF2videoresult.mp4'
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    detectimage.detectimagefolder_tovideo(folderpath, mydetector, outputvideopath)

if __name__ == "__main__":
    #Test TF2
    #testTF2Detector(TF2detectorargs)

    #Test Detectron2
    testDetectron2Detector(Detectron2detectorargs)

    #Test TorchVision
    #testTorchVisionDetector(TorchVisiondetectorargs)
