import scipy
#from imutils import face_utils
import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math
import playsound
from threading import Thread
from collections import OrderedDict


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
      
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def play_alarm(path):
   #playing sound file
   playsound.playsound(path)

def _square(co_1,co_2):
   sqr = math.pow((co_2-co_1),2)
   return sqr

#function calculating ecludian distance
def _ecludian_dist(s1,s2):
   return math.sqrt((s1+s2))

#function for calculatig eye aspect ratio
def Eye_Aspect_Ratio(eye):
   #vertical ecludian distance between pt. 1 & 5 and pt. 2 & 4
  # d1 = scipy.spatial.distance.ecludian(eye[1], eye[5])
   #d2 = scipy.spatial.distance.ecludian(eye[2], eye[4])
   sqr1 = _square(eye[1][1],eye[5][1])
   sqr2 = _square(eye[1][0],eye[5][0])
   dist1 = _ecludian_dist(sqr1,sqr2)  

   sqr1 = _square(eye[2][1],eye[4][1])
   sqr2 = _square(eye[2][0],eye[4][0])
   dist2 = _ecludian_dist(sqr1,sqr2)

   #dist1 and dist1 are verical distances
   
   #horiontal ecludian distance between pt. 0 & 3
   sqr1 = _square(eye[0][1],eye[3][1])
   sqr2 = _square(eye[0][0],eye[3][0])
   dist3 = _ecludian_dist(sqr1,sqr2)

   #eye aspect ratio
   eyeAratio = (dist1+dist2)/(2.0 * dist3)

   return eyeAratio

def mouth_width(_mouth_):
   sqr1 = _square(_mouth_[0][1],_mouth_[6][1])
   sqr2 = _square(_mouth_[0][0],_mouth_[6][0])

   return _ecludian_dist(sqr1,sqr2)


#-----------------------------------------------------------------------------------------------#
#---------------------------------Program Start Here--------------------------------------------#

ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True , help="fecial landmark detector path")
#ap.add_argument("-i","--picamera",default = -1,type = int, help="image path")
ap.add_argument("-a", "--alarm", required= True,help="path to sound file")
args = vars(ap.parse_args())

ear_threshold = 0.28
ear_consec_frames = 12

cnt = 0
total = 0
is_alarm = False;

(leyestrt, leyeend) = FACIAL_LANDMARKS_IDXS["left_eye"]
(reyestrt, reyeend) = FACIAL_LANDMARKS_IDXS["right_eye"]
(mthStart,mthEnd) = FACIAL_LANDMARKS_IDXS["mouth"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("Loading facial landmarks...")
cap = cv2.VideoCapture(0)
frame_cnt = 0
#image = cv2.imread(args["image"])
while(True):
    #image = imutils.resize(image, width=500)
    Avg_ear = 0
    ret, image = cap.read()
    frame_cnt +=1
    #time = time.time()
   # cv2.putText(image, "time: {}".format(time), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Output_gray",gray)
    rects = detector(gray,1)
    shape = None
    for(i,rect) in enumerate(rects):
        shape = predictor(gray,rect)
        shape = shape_to_np(shape)

        (x,y,w,h) = rect_to_bb(rect)

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Output_color",image)
        for(x,y) in shape:
            cv2.circle(image, (x,y),1,(0,0,255),-1)
        leftEye = shape[leyestrt:leyeend] #left eye co-ordinates
        rightEye = shape[reyestrt:reyeend] #right eye co-ordinates
        mthEndCords = shape[mthStart:mthEnd] #mouth co-ordinates

        #Eye aspect ratio calculations
        leftEAR = Eye_Aspect_Ratio(leftEye)
        rightEAR = Eye_Aspect_Ratio(rightEye)

        Avg_ear = (leftEAR + rightEAR)/2.0

        mthWidth = mouth_width(mthEndCords)
        #cv2.putText(image, "Horizanta mouth width: {}".format(mthWidth), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        leftCovHull = cv2.convexHull(leftEye)
        rightCovHull = cv2.convexHull(rightEye)
        cv2.drawContours(image, [leftCovHull], -1,(0,255,0),1)
        cv2.drawContours(image, [rightCovHull], -1,(0,255,0),1)

        cv2.putText(image, "EAR: {:.2f}".format(Avg_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Frame: {}".format(frame_cnt), (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mthWidth <=65:
           if Avg_ear < ear_threshold:
               cnt += 1
           #else:
               #if cnt>=ear_consec_frames:
                   #total += 1
                   #cnt = 0 
               if cnt>=ear_consec_frames:
                  if not is_alarm:
                     is_alarm = True

                     #chck whether path is blank
                     if args["alarm"]!= "":
                        thrd = Thread(target = play_alarm, args = (args["alarm"],))
                        thrd.daemon = True
                        thrd.start()
                     
                  cv2.putText(image, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
           else:
            cnt = 0;
            is_alarm = False
            
        #cv2.putText(image, "Blinks: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
    #output = face_utils.visualize_facial_landmarks(image, shape)
    #output = face_utils.visualize_facial_landmarks(image, shape)
    #cv2.imshow("Image",output)
    cv2.imshow("Output_color",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
#cv2.stop()
#cv2.waitKey(0)

#Liabraries\shape_predictor_68_face_landmarks.dat
#images/pic1.JPG
#python facial_landmark_detector.py --shape-predictor Liabraries\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat --alarm sound_files\alarm4.wav

