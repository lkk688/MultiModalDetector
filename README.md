# MultiModalDetector
This project creates a multimodal object detector via multiple state-of-the-art deep learning frameworks. 

# Trainer

## Object Detection training and evaluation based on Tensorflow2 Object Detection
* Tensorflow2-objectdetection-xxx.ipynb is the Google Colab sample code to perform object detection and training based on Tensorflow2 object detection (latest version) and utilize the converted Waymo TFRecord file in Google Cloud storage.

## Object Detection training and evaluation based on Pytorch Torchvision (FasterRCNN)
* The sample code to play with Torchvision in Colab: [colab link](https://colab.research.google.com/drive/1DKZUL5ylKjiKtfOCGpirjRA3j8rIOs9M?usp=sharing) (you need to use SJSU google account to view)
* WaymoTrain.ipynb is the sample code to perform object detection training based on Torchvision FasterRCNN based on the original Waymo Dataset (TFRecord format), no dataset conversion is used in this sample code
* WaymoEvaluation.ipynb is the sample code to perform evaluation (both COCO evaluation and image visualization) of the trained object detection model based on Torchvision FasterRCNN
* coco_eval.py, coco_utils.py, engine.py, transforms.py, utils.py are copied from Torchvision directory and used in the WaymoTrain.ipynb and WaymoEvaluation.ipynb

## Object Detection training and evaluation based on Detectron2
* WaymoDetectron2Train.py is the code to run training based on Detectron2. This code used the COCO formated dataset (WaymoDataset converted to COCO via WaymoNewtoCOCO.ipynb)
* WaymoDetectron2Evaluation.ipynb is the jupyter notebook code to run evaluation based on Detectron2

## Object Detection training and evaluation based on mmdetection2d
* mymmdetection2dtrainxx.py is the training code based on mmdetection2d
* mymmdetection2d.py is the evaluation code for models from mmdetection2d


# MyDetector
MyDetector folder is the generalized inference (Detector) code for object detection based on multiple frameworks (Tensorflow2, Detectron2, mmdetection2d, and Pytorch torchvision)

# MyTracker
MyTracker is the tracking code for the bounding box output after the detection module.
