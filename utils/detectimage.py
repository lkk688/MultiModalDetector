#from utils.plotresults import show_image_bbxyxy
import matplotlib
import os
import time
import cv2
import glob
from pathlib import Path


import importlib
from utils import plotresults
importlib.reload(plotresults)#only used for debug


def detectoneimage(imgpath, mydetector):
    start = time.time()
    ori_im = cv2.imread(imgpath)
    imageshape=ori_im.shape
    im_width=imageshape[1]#2720#800
    im_height=imageshape[0]#1530#600
    print("Image width: ", im_width)
    print("Image height: ", im_height)

    im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
    bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)

    end = time.time()
    #show_image_bbxyxy(im, bbox_xyxy, cls_ids, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    #cv2.imwrite('resulttest.jpg', imresult) 
    print("time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))

def detectoneimage_novis(imgpath, mydetector):
    start = time.time()
    ori_im = cv2.imread(imgpath)
    if ori_im is None:
        print("Image not available!")
        return [], [], []
    imageshape=ori_im.shape
    im_width=imageshape[1]#2720#800
    im_height=imageshape[0]#1530#600
    print("Image width: ", im_width)
    print("Image height: ", im_height)

    im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
    bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)

    end = time.time()
    #show_image_bbxyxy(im, bbox_xyxy, cls_ids, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    #plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, 'testimg.pdf')
    img_box = plotresults.draw_boxes(im, bbox_xyxy, cls_ids, cls_conf, mydetector.FULL_LABEL_CLASSES)
    cv2.imwrite('resulttest.jpg', img_box) 
    print("Detection time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))
    print("Image result saved to resulttest.jpg")

    #pred_labels = [mydetector.FULL_LABEL_CLASSES[i] for i in list(cls_ids) ]
    return bbox_xyxy, cls_ids, cls_conf

def detectimagefolder_tovideo(imgpath, mydetector, outputvideopath):
    imagepath=sorted(glob.glob(imgpath+'/*.jpg'))
    imglen=len(imagepath)
    print("Total image:", imglen)
    imgidx=0
    results = []
        
    first_im = cv2.imread(imagepath[0])
    imageshape=first_im.shape
    im_width=imageshape[1]#2720#800
    im_height=imageshape[0]#1530#600
    print("Image width: ", im_width)
    print("Image height: ", im_height)
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
 #cv2.VideoWriter_fourcc(*'MJPG')#cv2.VideoWriter_fourcc('M','P','4','V')  #cv2.VideoWriter_fourcc(*'MJPG')
    videooutput = cv2.VideoWriter(outputvideopath, fourcc, 1, (im_width, im_height)) #20
        
    for imgidx in range(imglen): #imglen  while self.vdo.grab():
        filepath=imagepath[imgidx]
        f_image_path=Path(filepath)#convert str to Path object
        f_image_name = f_image_path.name
        print(f_image_name)#image file name
        imgfiles=os.path.splitext(f_image_name)
        imageid=imgfiles[0] #image filename without .jpg

        start = time.time()
        #_, im = self.vdo.retrieve()
        ori_im = cv2.imread(filepath)
        print(ori_im.shape)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        #Perform detection
        bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)
        end = time.time()
        print("Detection time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))

        img_box = plotresults.draw_boxes(im, bbox_xyxy, cls_ids, cls_conf, mydetector.FULL_LABEL_CLASSES)
        videooutput.write(img_box)
        #plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, outputvideopath+imageid+''.jpg')
    
    videooutput.release()
    print("Finished all detections")
