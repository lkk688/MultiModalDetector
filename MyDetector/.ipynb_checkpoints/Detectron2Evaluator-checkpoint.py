import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

import copy
import logging
from typing import Any, Dict, Tuple
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode

#from densepose import DatasetMapper, DensePoseCOCOEvaluator, add_densepose_config
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper

from detectron2.data import detection_utils as utils

from fvcore.transforms.transform import TransformList, Transform, NoOpTransform

import os
import numpy as np
#%matplotlib inline
#from matplotlib import pyplot as plt

class Detectron2COCOEvaluator(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
    #args.datasetname
    #args.FULL_LABEL_CLASSES
    #self.args.datasetpath="/data/cmpe295-liu/Waymo/WaymoCOCOsmall/"

    def __init__(self, args):
        self.args = args
        use_cuda = True
        
#         for d in ["Training", "Validation"]:
#             DatasetCatalog.register("mywaymo1_" + d, lambda d=d: load_coco_json(self.args.datasetpath + d + "/annotations.json", self.args.datasetpath + d + "/"))

        d="Validation"
        self.datasetname=args.datasetname+d
        allregistereddataset=DatasetCatalog.list()#return List[str]
        print(allregistereddataset)
        if self.datasetname not in allregistereddataset:
            jsonpath=os.path.join(self.args.datasetpath, d, 'annotations.json')
            DatasetCatalog.register(self.datasetname, lambda d=d: load_coco_json(jsonpath, os.path.join(self.args.datasetpath, d)))
            print("Registered dataset: ", self.datasetname)
        
        #FULL_LABEL_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']#['ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus',  'motor', 'others']
        MetadataCatalog.get(self.datasetname).set(thing_classes=args.FULL_LABEL_CLASSES)
#         for d in ["Training", "Validation"]:
#             MetadataCatalog.get("mywaymo1_" + d).set(thing_classes=FULL_LABEL_CLASSES)
        
        cfg = get_cfg()
        #cfg.OUTPUT_DIR='./output' #'./output_x101'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+self.args.modelname+".yaml" ))#faster_rcnn_X_101_32x8d_FPN_3x
        #cfg.DATASETS.TRAIN = ("mywaymo1_Training",)
        cfg.DATASETS.TEST = (self.datasetname,)
        cfg.DATALOADER.NUM_WORKERS = 1 #2
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
        #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local 
        #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output', "model_final.pth")
        cfg.MODEL.WEIGHTS = os.path.join(self.args.modelbasefolder, self.args.modelfilename) #model_0159999.pth
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.LR_SCHEDULER_NAME='WarmupCosineLR'
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 80000# 140000    # you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #512#128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(args.FULL_LABEL_CLASSES) #5 #12  # Kitti has 9 classes (including donot care)

        cfg.TEST.EVAL_PERIOD = 5000
        
        self.cfg_detectron2=cfg

        self.predictor = DefaultPredictor(cfg)

    def evaluate(self):
        #evaluator = COCOEvaluator(self.datasetname, self.cfg_detectron2, False, output_dir=outputpath)
        evaluator = COCOEvaluator(self.datasetname, self.cfg_detectron2, False, output_dir=self.args.modelbasefolder)
        val_loader = build_detection_test_loader(self.cfg_detectron2, self.datasetname)
        inference_on_dataset(self.predictor.model, val_loader, evaluator)

#         bbox_xcycwh, cls_conf, cls_ids = [], [], []

#         #box format is XYXY_ABS
#         for (box, _class, score) in zip(boxes, classes, scores):
#             #if _class == 0: # the orignal code only track people?
#             x0, y0, x1, y1 = box
#             bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)]) # convert to x-center, y-center, width, height
#             cls_conf.append(score)
#             cls_ids.append(_class)

#         return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)