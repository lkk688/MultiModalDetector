import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import os
import argparse
from pathlib import Path
import cv2
import json
#import utils
from PIL import Image
from glob import glob
import sys
import datetime
import os

WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

def get_camera_labels(frame):
    if frame.camera_labels:
        return frame.camera_labels
    return frame.projected_lidar_labels

def extract_segment_allfrontcamera(PATH,folderslist, out_dir, step):
    
    #folderslist = ["training_0031","training_0030","training_0029","training_0028","training_0027","training_0026"]
    #PATH='/data/cmpe295-liu/Waymo'
    images = []
    annotations = []
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]
    image_globeid=0
        
    for index in range(len(folderslist)):
        foldername=folderslist[index]
        print("Folder name:", foldername)
        tfrecord_files = glob(os.path.join(PATH, foldername, "*.tfrecord")) #[path for path in glob(os.path.join(PATH, foldername, "*.tfrecord"))]
        print("Num of tfrecord file:", len(tfrecord_files))
        #print(tfrecord_files)
    
        for segment_path in tfrecord_files:

            print(f'extracting {segment_path}')
            segment_path=Path(segment_path)#convert str to Path object
            segment_name = segment_path.name
            print(segment_name)
            segment_out_dir = out_dir # remove segment_name as one folder, duplicate with image name
        #     segment_out_dir = out_dir / segment_name 
        #     print(segment_out_dir)#output path + segment_name(with tfrecord)
        #     segment_out_dir.mkdir(parents=True, exist_ok=True)

            dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')

            for i, data in enumerate(dataset):
                if i % step != 0:
                    continue

                print('.', end='', flush=True)
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                #get one frame

                context_name = frame.context.name
                frame_timestamp_micros = str(frame.timestamp_micros)

                for index, image in enumerate(frame.images):
                    if image.name != 1: #Only use front camera
                        continue
                    camera_name = open_dataset.CameraName.Name.Name(image.name)
                    image_globeid = image_globeid + 1
                    #print("camera name:", camera_name)

                    img = tf.image.decode_jpeg(image.image).numpy()
                    image_name='_'.join([frame_timestamp_micros, camera_name])#image name
                    #image_id = '/'.join([context_name, image_name]) #using "/" join, context_name is the folder
                    #New: do not use sub-folder
                    image_id = '_'.join([context_name, image_name])
                    #image_id = '/'.join([context_name, frame_timestamp_micros, camera_name]) #using "/" join
                    
                    file_name = image_id + '.jpg'
                    #print(file_name)
                    file_name = '/'.join([foldername, file_name])
                    filepath = out_dir / file_name
                    #filepath = segment_out_dir / file_name
                    #print('Image output path',filepath)
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    #images.append(dict(file_name=file_name, id=image_id, height=img.shape[0], width=img.shape[1], camera_name=camera_name))#new add camera_name
                    images.append(dict(file_name=file_name, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name))#new add camera_name
                    #print("current image id: ", image_globeid)
                    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#OpenCV always deal with BGR
                    cv2.imwrite(str(filepath), img)

                    for camera_labels in get_camera_labels(frame):
                        # Ignore camera labels that do not correspond to this camera.
                        if camera_labels.name == image.name:
                            # Iterate over the individual labels.
                            for label in camera_labels.labels:
                                # object bounding box.
                                width = int(label.box.length)
                                height = int(label.box.width)
                                x = int(label.box.center_x - 0.5 * width)
                                y = int(label.box.center_y - 0.5 * height)
                                area = width * height
                                annotations.append(dict(image_id=image_globeid,
                                                        bbox=[x, y, width, height], area=area, category_id=label.type,
                                                        object_id=label.id,
                                                        tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                        detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1))

    with (segment_out_dir / 'annotations.json').open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #set as image frame ID
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)

def extractWaymoTFrecordimages(PATH, folderslist, out_dir, step):
    #PATH='/mnt/DATA5T/WaymoDataset'
    #folderslist = validation_folders = ["Validation0000"] #["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005", "validation_0006", "validation_0007"]
    tfrecord_files = [path for x in folderslist for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
    print(len(tfrecord_files))#total number of tfrecord files
    #out_dir='/mnt/DATA5T/WaymoDataset/Extracted'
    #step=1 #downsample
    out_dir = Path(out_dir)
    extract_segment_allfrontcamera(PATH,folderslist, out_dir, step)