import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import numpy as np

from yolov5models.experimental import attempt_load

from yolov5models.yolo import Model

# from ultralyticsyolov5 import *

class ultralyticsYolov5Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # Load model
        print(os.getcwd())
        weightpath=os.path.join(args.modelbasefolder, args.modelfilename)
        # Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)#.fuse().eval()  # yolov5s.pt
        #self.model = self.model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

    def detect(self, im):
        self.testyolo5()
    
    def testyolo5(self):
        # Images
        for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
            torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
        img1 = Image.open('zidane.jpg')  # PIL image
        img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
        img3 = np.zeros((640, 1280, 3))  # numpy array
        imgs = [img1, img2, img3]  # batched list of images

        # Inference
        with torch.no_grad():
            prediction = self.model(imgs, size=640)  # includes NMS

        # Plot
        for i, (img, pred) in enumerate(zip(imgs, prediction)):
            str = 'Image %g/%g: %gx%g ' % (i + 1, len(imgs), *img.shape[:2])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
                    # str += '%s %.2f, ' % (label, conf)  # label
                    ImageDraw.Draw(img).rectangle(box, width=3)  # plot
            img.save('results%g.jpg' % i)  # save
            print(str + 'Done.')

class MyYolov5Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename

    def __init__(self, args):
        self.args = args
        self.device = args.device
        # Load model
        print(os.getcwd())
        weightpath=os.path.join(args.modelbasefolder, args.modelfilename)

        model = attempt_load(weightpath, map_location=self.device)  # load FP32 model
        self.model = Model('./MyDetector/yolov5models/yolov5s.yaml').to(self.device)
        self.model.load_state_dict(torch.load(weightpath))
        self.model = self.model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS


        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device).eval()
        #self.model = attempt_load(weightpath, map_location=self.device)  # load FP32 model
        # Get names and colors
        #names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


        #self.FULL_LABEL_CLASSES=args.FULL_LABEL_CLASSES
        use_cuda = True
        self.threshold = args.threshold if args.threshold is not None else 0.1

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        self.testyolo5()
        img = torch.from_numpy(im).to(self.device)
        #img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        predall=self.model(img)
        pred = self.model(img)[0]
        #return outputs
        print(pred)
        return pred #np.array(pred_boxes), np.array(pred_class), np.array(pred_score)


    def testyolo5(self):
        # Images
        for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
            torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
        img1 = Image.open('zidane.jpg')  # PIL image
        img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
        img3 = np.zeros((640, 1280, 3))  # numpy array
        imgs = [img1, img2, img3]  # batched list of images

        # Inference
        with torch.no_grad():
            prediction = self.model(imgs, size=640)  # includes NMS

        # Plot
        for i, (img, pred) in enumerate(zip(imgs, prediction)):
            str = 'Image %g/%g: %gx%g ' % (i + 1, len(imgs), *img.shape[:2])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
                    # str += '%s %.2f, ' % (label, conf)  # label
                    ImageDraw.Draw(img).rectangle(box, width=3)  # plot
            img.save('results%g.jpg' % i)  # save
            print(str + 'Done.')
