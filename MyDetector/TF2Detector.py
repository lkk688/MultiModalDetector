import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
from object_detection.builders import model_builder


#%matplotlib inline
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

class MyTF2Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
    #args.labelmappath

    def __init__(self, args):
        self.args = args
        #self.FULL_LABEL_CLASSES=args.FULL_LABEL_CLASSES
        self.threshold = args.threshold
        
        tf.keras.backend.clear_session()
        self.detect_fn = tf.saved_model.load(args.modelbasefolder)
        
        label_map_path=args.labelmappath #'./models/research/object_detection/data/mscoco_label_map.pbtxt'
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        self.FULL_LABEL_CLASSES=list(label_map_dict.keys())
        

    def detect(self, im):
        imageshape=im.shape
        im_width=imageshape[1]#2720#800
        im_height=imageshape[0]#1530#600
    
        input_tensor = np.expand_dims(im, 0)
        detections = self.detect_fn(input_tensor)
        
        #[0] means get the first batch, only one batch, 
        boxes = detections['detection_boxes'][0].numpy() #xyxy type [0.26331702, 0.20630795, 0.3134004 , 0.2257505 ], [ymin, xmin, ymax, xmax]
        #print(boxes)
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        #print(classes)
        scores = detections['detection_scores'][0].numpy() #decreasing order
        
        #predlist=[scores.index(x) for x in scores if x > self.threshold] # Get list of index with score greater than threshold.
        pred_score=[x for x in scores if x > self.threshold] # Get list of index with score greater than threshold.
        pred_t = np.where(scores==pred_score[-1])#get the last index
        #print(pred_t)
        pred_t=pred_t[0][0] #fetch value from tuple of array
        #print(pred_t)
        print("Box len:", len(boxes))
        pred_boxes = boxes[:pred_t+1]
        print("pred_score len:", len(pred_score))
        #print("pred_boxes len:", len(pred_boxes))
        pred_class = classes[:pred_t+1]
        pred_class = [i-1 for i in list(pred_class)] # index starts with 1, 0 is the background in the tensorflow
        #print(pred_class)
            
        #[ (xmin, ymin), (xmax, ymax)]
        pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes
        
        return pred_boxes, pred_class, scores
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