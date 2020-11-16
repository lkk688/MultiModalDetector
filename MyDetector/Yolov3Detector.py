from yolov3.models import *

import glob
import math
import os
import random
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class MyYolov3Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
    #labelmappath='MyDetector/yolov3/data/coco.names'
    #img_size=512
    #threshold = 0.3
    #iou_threshold = 0.6

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.threshold = args.threshold
        self.iou_threshold = args.iou_threshold
        
        # Load model
        print(os.getcwd())
        # Initialize model
        imgsz = (320, 192) # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.model = Darknet(args.modelconfig, imgsz)

        # Load weights
        weights=os.path.join(args.modelbasefolder, args.modelfilename)#weightpath
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, weights)
        
        # Eval mode
        self.model.to(self.device).eval()

        # Get names and colors
        self.names = load_classes(args.labelmappath)#80 classes
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, im):
        img = np.transpose(im,(2,0,1))#change a numpy HxWxC array to CxHxW, ARRAY
        img = torch.from_numpy(img).to(self.device) #Tensor [1280, 1920, 3]
        half = False
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) #[3, 288, 512]->[1, 3, 288, 512]

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0] #torch.Size([1, 151200, 85])

        # Apply NMS
        pred = non_max_suppression(pred, self.threshold, self.iou_threshold, multi_label=False) #6 element
        
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        
        return pred
        
    