import argparse
import glob
import os
import time
import cv2

from MyDetector import TF2Detector
from MyDetector import Detectron2Detector
from MyDetector import TorchVisionDetector
from Myutils import detectandtrack, detectimage

from MyTracker.deep_sort import DeepSort
from MyTracker.deep_sort import MyDeepSort
from Myutils import extractWaymoTFrecordimages

from MyTracker.motevaluation import MoTEvaluator
import motmetrics as mm

class Detectron2detectorargs:
    modelname = 'faster_rcnn_X_101_32x8d_FPN_3x'
    modelbasefolder = '../ModelOutput'
    #modelfilename='waymo_fasterrcnnx101_detectron2model_final.pth' #
    modelfilename='fasterrcnnx101.pth' #
    showfig='True'
    #FULL_LABEL_CLASSES=['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    FULL_LABEL_CLASSES=['person', 'bicycle', 'car', 'motorcycle' , 'airplane' , 'bus']
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

def waymotrackingdemo():
    #Extract the waymo tf record file to image folder
    PATH='/mnt/DATA5T/WaymoDataset'
    folderslist = validation_folders = ["Validation0000"]
    out_dir='/mnt/DATA5T/WaymoDataset/Extracted'
    step=1
    extractWaymoTFrecordimages.extractWaymoTFrecordimages(PATH, folderslist, out_dir, step)

    deepsort_checkpoint='/home/kaikai/Documents/MyDetector/ModelOutput/deepsortcheckpoint/ckpt.t7'
    mydeepsort = MyDeepSort(deepsort_checkpoint, use_cuda=True)

    mydetector = Detectron2Detector.MyDetectron2Detector(Detectron2detectorargs)

    outputvideopath='detectron2trackvideoresult.mp4'
    #folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/WaymoCOCO/Validation/', 'validation_0000')
    folderpath=os.path.join('/mnt/DATA5T/WaymoDataset/Extracted', 'Validation0000')
    detectandtrack.trackimagefolder_tovideo(folderpath, mydetector, mydeepsort, outputvideopath)

def videotrackingdemo():
    deepsort_checkpoint='/home/kaikai/Documents/MyDetector/ModelOutput/deepsortcheckpoint/ckpt.t7'
    mydeepsort = MyDeepSort(deepsort_checkpoint, use_cuda=True)

    mydetector = Detectron2Detector.MyDetectron2Detector(Detectron2detectorargs)

    outputvideopath='../results/nvidia_s01c004.mp4'
    #videopath=os.path.join('/mnt/DATA5T/SJSUVideo/03212019', '0120190321151319.mp4')
    videopath=os.path.join('/mnt/DATA5T/NVIDIAAICitydataset/Track1/train/S01/c004/', 'vdo.avi')#rlc2-151808-152038
    detectandtrack.trackvideo_tovideo(videopath, mydetector, mydeepsort, outputvideopath)

def motevaluation():
    data_root='/mnt/DATA5T/MOT/MOT17/train'
    result_path='/mnt/DATA5T/MOT/results'
    seqs=('MOT17-02-SDP','MOT17-04-SDP', 'MOT17-05-SDP',)
    #seq='MOT17-02-SDP'
    data_type = 'mot'
    accs = []
    for seq in seqs:
        evaluator = MoTEvaluator(data_root, seq, data_type)
        result_filename=os.path.join(result_path, '{}.txt'.format(seq))
        #result_filename=os.path.join(result_root, '{}.txt'.format(seq))
        accs.append(evaluator.eval_file(result_filename))
        #evaluator.eval_file(result_filename)

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = MoTEvaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    MoTEvaluator.save_summary(summary, os.path.join('../results', 'summary_{}.xlsx'.format('mot')))

if __name__ == "__main__":
    #waymotrackingdemo()
    videotrackingdemo()
    #motevaluation()

    