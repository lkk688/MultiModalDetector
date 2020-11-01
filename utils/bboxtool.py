import numpy as np

def _xyxy_to_tlwh(bbox_xyxy):
    x1,y1,x2,y2 = bbox_xyxy

    t = x1
    l = y1
    w = int(x2-x1)
    h = int(y2-y1)
    return t,l,w,h

#convert bbox_xyxy ([ (xmin, ymin), (xmax, ymax)] tuple) to numpy.ndarray bbox_xcycwh
def _xyxy_to_xcycwh(bbox_xyxy):
    #bbox_xyxy=np.array(bbox_xyxy)#convert to nparray
    bbox_xcycwh=[]
    boxnum=len(bbox_xyxy)
    for i in range(boxnum):#patch in pred_bbox:
        patch=bbox_xyxy[i] # [ (xmin, ymin), (xmax, ymax)]
        #patch[0] (xmin, ymin)
        #patch[1] (xmax, ymax)
        x0=int(patch[0][0])
        y0=int(patch[0][1])
        x1=int(patch[1][0])
        y1=int(patch[1][1])
        bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)]) # convert to x-center, y-center, width, height
    return np.array(bbox_xcycwh)#convert to nparray