from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

import os
import numpy as np
#%matplotlib inline
#from matplotlib import pyplot as plt
#from MyDetector.Postprocess import postfilter
from MyDetector import Postprocess

class MyDetectron2Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename

    def __init__(self, args):
        self.args = args
        self.FULL_LABEL_CLASSES=args.FULL_LABEL_CLASSES
        use_cuda = True
        self.threshold = args.threshold if args.threshold is not None else 0.1
#         self.cfg = get_cfg()
#         self.cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#         self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#         self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
#         self.predictor = DefaultPredictor(self.cfg)
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+self.args.modelname+".yaml" ))#faster_rcnn_X_101_32x8d_FPN_3x
        #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))#faster_rcnn_X_101_32x8d_FPN_3x
        #cfg.merge_from_file('faster_rcnn_R_101_C4_3x.yaml')#Tridentnet
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        #cfg.DATASETS.TRAIN = ("myuav1_train",)
        #cfg.DATASETS.TEST = ("myuav1_val",)
        self.cfg.DATALOADER.NUM_WORKERS = 1 #2
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
        self.cfg.MODEL.WEIGHTS = os.path.join(self.args.modelbasefolder, self.args.modelfilename) #model_0159999.pth
        if os.path.isfile(self.cfg.MODEL.WEIGHTS) == False:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/"+self.args.modelname+".yaml")  # Let training initialize from model zoo
        else:
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512#128   # faster, and good enough for this toy dataset (default: 512)
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.FULL_LABEL_CLASSES)  # Kitti has 9 classes (including donot care)
        #self.cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output_uav', "model_0119999.pth") #model_0159999.pth
        #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local downloaded model
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LRgkyjmh,
        self.cfg.SOLVER.MAX_ITER = 100000    # you may need to train longer for a practical dataset

        self.cfg.TEST.DETECTIONS_PER_IMAGE = 500
        
        self.predictor = DefaultPredictor(self.cfg)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        outputs = self.predictor(im)
        #return outputs
        #print(outputs)
        
        #if (self.args.showfig):
#         plt.figure(figsize=(20,30))
#         v = Visualizer(im[:, :, ::-1],
#                        metadata=uavval_metadata, 
#                        scale=0.8,  )
#         v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         #cv2_imshow(v.get_image()[:, :, ::-1])
#         plt.imshow(v.get_image()[:, :, ::-1])
#         plt.show()
            
#             predictions = outputs["instances"].to("cpu")
#             boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
#             newboxes=_convert_boxes(boxes)
            #bbox_xcycwh=BoxMode.convert(newboxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)  #BoxMode.convert(x, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_class = outputs["instances"].pred_classes.cpu().numpy()
        pred_score = outputs["instances"].scores.cpu().numpy()
        
        #bbox_xywh=BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        #pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred_boxes)] # Bounding boxes
        #pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        
        #Post filter based on threshold
        #pred_boxes, pred_class, pred_score = postfilter(pred_boxes, pred_class, pred_score, self.threshold)
        pred_boxes, pred_class, pred_score = Postprocess.postfilter_thresholdandsize(pred_boxes, pred_class, pred_score, self.threshold, minsize=1)
        
        #return pred_boxes, pred_class, pred_score
        return np.array(pred_boxes), np.array(pred_class), np.array(pred_score)


#         bbox_xcycwh, cls_conf, cls_ids = [], [], []

#         #box format is XYXY_ABS
#         for (box, _class, score) in zip(boxes, classes, scores):
#             #if _class == 0: # the orignal code only track people?
#             x0, y0, x1, y1 = box
#             bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)]) # convert to x-center, y-center, width, height
#             cls_conf.append(score)
#             cls_ids.append(_class)

#         return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)

#     def detectwithvisualization(self, im)
#         pred_boxes, classes, scores=self.detect(im)