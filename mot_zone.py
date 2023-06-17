#initializing flat settings for yolo
from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)


import time #frame per second
import numpy as np 
import cv2 #visualizing tracking
import matplotlib.pyplot as plt #color map

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching #setting up deep association matrix
from deep_sort.detection import Detection #detect object
from deep_sort.tracker import Tracker #track info
from tools import generate_detections as gdet #import features generation

#initialize Yolo, load class names and weights
class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

#initialize deepsort including parameters and encode functions
max_cosine_distance = 0.5 #considering object is same or not. Larger number 
nn_budget = None #form gallery for each detection by using deep network. Default is 100
nms_max_overlap = 0.8 #avoid too many detection for same object. Default 1

#initialzie encoder
model_filename ='model_data/mars-small128.pb' #tracking pedestrian model
encoder = gdet.create_box_encoder(model_filename, batch_size = 1) 
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

#capture video and assign output
vid = cv2.VideoCapture('./data/video/testcar.mp4')
#assign output
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) ,int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/zone_results.avi', codec, vid_fps, (vid_width, vid_height))

#create a deque 
from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

#count everything
counter = []

#capture all frames from videos
while True:
    _, img = vid.read() #read frame one by one
    if img is None:
        print('Completed')
        break
    
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change color format bgr to rgb
    img_in = tf.expand_dims(img_in,0) #expand dimension. Image has 3d array shape, add 1 more dimension batch size
    img_in = transform_images(img_in,416) #resize/reshape image for yolov3. default is 416
    
    #yolo predictions
    t1 = time.time()
    
    boxes, scores, classes, nums = yolo.predict(img_in) #pass image to yolo prediction function
    
    # boxes, 3d shape (1,100,4) max 100 boxes per image 
    # scores, 2d shape (1,100) 
    # classes, 2d shape (1,100) detected object classes
    # nums, 1d shape (1) total number of detected objects
    
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0]) #convert boxes into a list
    features = encoder(img, converted_boxes) #generate features for each detected objects
    
    detections = [Detection(bbox,score,class_name,feature) for bbox,score,class_name,feature in 
                  zip(converted_boxes,scores[0], names, features)] 
    
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    
    indices = preprocessing.non_max_suppression(boxs,classes,nms_max_overlap,scores) 
    
    detections = [detections[i] for i in indices] #to eliminate multiple frame on one target
    
    tracker.predict()
    tracker.update(detections) 
    
    #visualise the results
    cmap = plt.get_cmap('tab20b') #create color map
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)] #assign colors in color mapping
    
    #count detected objects within area
    current_count = int(0)
    
    #loop results from tracker 
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1: #if filtering no track and no update
            continue
        
        bbox = track.to_tlbr() 
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)] #get remainders for assigning color code
        color = [i * 255 for i in color] #to change to standard rgb scale
        
        #create rectangle 
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2) 
        #rectangle above box to show length and track id
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1) 
        #put text into above rectangle
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)
        
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        
        #create historical trajectory line
        for j in range(1,len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            
            thickness = int(np.sqrt(64/float(j+1))*2) #keep closer thin and far keep thick
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness) #draw line
        
        #draw line   
        height, width, _ = img.shape
        cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6)), (0, 255, 0), thickness = 2)
        cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
       
        center_y = int(((bbox[1])+ (bbox[3]))/2)
        
        if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
            if class_name == 'car' or class_name == 'truck' or class_name == 'motorcycle':
                counter.append(int(track.track_id))
                current_count += 1
                
                
    total_count = len(set(counter))
    cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img,'Total vehicle count: ' + str(total_count), (0,130), 0,1,(0,0,255), 2 )

        
    #print fps
    fps = 1./(time.time()-t1)
    cv2.putText(img,"FPS: {:.2f}".format(fps),(0,30),0,1,(0,0,255),2)
    cv2.resizeWindow('output',1024,768)
    cv2.imshow('output',img)
    out.write(img)
    
    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()
