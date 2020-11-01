import numpy as np

def postfilter(boxes, classes, scores, threshold):
    pred_score=[x for x in scores if x > threshold] # Get list of score with score greater than threshold.
    #print(pred_score)
    if len(pred_score)<1:
        print("Empty")
        pred_boxes=[]
        pred_class=[]
        pred_score=[]
    else:
        pred_t = np.where(scores==pred_score[-1])#get the last index
        #print(pred_t)
        pred_t=pred_t[0][0] #fetch value from tuple of array, (array([2]),)
        #print(pred_t)
        print("Box len:", len(boxes))
        pred_boxes = boxes[:pred_t+1]
        print("pred_score len:", len(pred_score))
        #print("pred_boxes len:", len(pred_boxes))
        pred_class = classes[:pred_t+1]
    return pred_boxes, pred_class, pred_score

def postfilter_thresholdandsize(boxes, classes, scores, threshold, minsize):
    num=len(scores)
    pred_boxes=[]
    pred_class=[]
    pred_score=[]
    for i in range(num):
        boxsizex=boxes[i][1][0]-boxes[i][0][0]
        boxsizey=boxes[i][1][1]-boxes[i][0][1]
        if scores[i]>threshold and boxsizex>minsize and boxsizey>minsize:
            pred_boxes.append(boxes[i])
            pred_class.append(classes[i])
            pred_score.append(scores[i])
    return pred_boxes, pred_class, pred_score