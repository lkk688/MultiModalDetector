import matplotlib
import os
import time
import cv2
import glob
import numpy as np
from pathlib import Path
from Myutils import plotresults
from Myutils import bboxtool
from Myutils import exportresults

def trackimagefolder_tovideo(imgpath, mydetector, mytracker, outputvideopath):
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
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)#opencv always read image in BGR format, convert to RGB format
        #Perform detection
        bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)
        end = time.time()
        print("Detection time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))

        #convert bbox_xyxy (tuple, [(189.2432, 646.61523), (298.20505, 718.8962)]) to bbox_xcycwh (numpy.ndarray, [243.5, 682.0, 109, 72]) 
        bbox_xcycwh=bboxtool._xyxy_to_xcycwh(bbox_xyxy)
        
        if bbox_xcycwh is not None and len(bbox_xcycwh)>0:
            #select classes
            mask = cls_ids <= 2
            cls_ids = cls_ids[mask]
            bbox_xcycwh = bbox_xcycwh[mask]
            #bbox_xcycwh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            deepoutputs = mytracker.update(bbox_xcycwh, cls_conf, cls_ids, im)
            # draw boxes for visualization
            if len(deepoutputs) > 0:
                bbox_tlwh = [] #same to xywh
                bbox_xyxy = deepoutputs[:, :4]
                identities = deepoutputs[:, -2]
                track_class = deepoutputs[:, -1]
                im = plotresults.draw_trackingboxes(im, bbox_xyxy, identities, track_class, mydetector.FULL_LABEL_CLASSES)
                #ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(bboxtool._xyxy_to_tlwh(bb_xyxy)) #convert to xywh
                    #bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy)) #convert to xywh

                results.append((imgidx, bbox_tlwh, identities))


        #img_box = plotresults.draw_boxes(im, bbox_xyxy, cls_ids, cls_conf, mydetector.FULL_LABEL_CLASSES)
        videooutput.write(im)
        #plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, outputvideopath+imageid+''.jpg')
    
    videooutput.release()
    print("Finished all detections")

def trackvideo_tovideo(video_path, mydetector, mytracker, outputvideopath):
    cap = cv2.VideoCapture(video_path)
    framerate=cap.get(cv2.CAP_PROP_FPS)
    im_width=int(cap.get(3))#2304
    im_height=int(cap.get(4))#1296
    print("Image width: ", im_width)
    print("Image height: ", im_height)

    results = []
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
 #cv2.VideoWriter_fourcc(*'MJPG')#cv2.VideoWriter_fourcc('M','P','4','V')  #cv2.VideoWriter_fourcc(*'MJPG')
    videooutput = cv2.VideoWriter(outputvideopath, fourcc, framerate, (im_width, im_height)) #20
    
    currentframe = 0
    while (cap.isOpened()):
        start = time.time()
        ret, ori_im = cap.read() #image: (1296, 2304, 3)
        if len((np.array(ori_im)).shape) == 0:
            break
        currentframe=currentframe+1
        print(ori_im.shape)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)#opencv always read image in BGR format, convert to RGB format
        #Perform detection
        bbox_xyxy, cls_ids, cls_conf = mydetector.detect(im)
        end = time.time()
        print("Detection time: {:.03f}s, fps: {:.03f}, detection numbers: {}".format(end - start, 1 / (end - start), len(bbox_xyxy)))

        trackstart = time.time()
        #convert bbox_xyxy (tuple, [(189.2432, 646.61523), (298.20505, 718.8962)]) to bbox_xcycwh (numpy.ndarray, [243.5, 682.0, 109, 72]) 
        bbox_xcycwh=bboxtool._xyxy_to_xcycwh(bbox_xyxy)
        
        if bbox_xcycwh is not None and len(bbox_xcycwh)>0:
            #select classes
            #dutclasses = [1,3,4,5,6,9]
            #mask=[(classi in dutclasses) for classi in cls_ids]
            mask = cls_ids <= 6
            cls_ids = cls_ids[mask]
            bbox_xcycwh = bbox_xcycwh[mask]
            #bbox_xcycwh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            deepoutputs = mytracker.update(bbox_xcycwh, cls_conf, cls_ids, im)
            trackend = time.time()
            print("Tracking time: {:.03f}s, fps: {:.03f}, track numbers: {}".format(trackend - trackstart, 1 / (trackend - trackstart), len(deepoutputs)))
            # draw boxes for visualization
            if len(deepoutputs) > 0:
                bbox_tlwh = [] #same to xywh
                bbox_xyxy = deepoutputs[:, :4]
                identities = deepoutputs[:, -2]
                track_class = deepoutputs[:, -1]
                im = plotresults.draw_trackingboxes(im, bbox_xyxy, identities, track_class, mydetector.FULL_LABEL_CLASSES)
                #ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(bboxtool._xyxy_to_tlwh(bb_xyxy)) #convert to xywh
                    #bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy)) #convert to xywh

                results.append((currentframe, bbox_tlwh, identities))
        end = time.time()
        print("time: {}s, fps: {}".format(end - start, 1/(end-start)))


        #img_box = plotresults.draw_boxes(im, bbox_xyxy, cls_ids, cls_conf, mydetector.FULL_LABEL_CLASSES)
        videooutput.write(im)
        #plotresults.show_imagewithscore_bbxyxy(im, bbox_xyxy, cls_ids, cls_conf, imgpath, mydetector.FULL_LABEL_CLASSES, outputvideopath+imageid+''.jpg')
    
    videooutput.release()
    exportresults.write_results('motresults.txt', results, 'mot')
    print("Finished all detections")