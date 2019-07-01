import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import csv
import pickle


#parametersim

EXPFOLDER="experiment"
impath=EXPFOLDER+'/data/imgs/'

filename="experiment.avi"
cap = cv2.VideoCapture(filename)
count=0
index=0
spacing=1/30 # seconds
fps=30 #movie framerate

#initializationparams

waitInit=2 #number of frames to wait before initializing tracking


#paramaters

bodythresh=200
body_area_min=20
body_area_max=800
body_circ=0.1

fgbgHISTORY=100
fgbgTHRESHOLD=32
fgbgHISTORYfast=5

edgeMIN=50
edgeMAX=250

dilateITER=1
erodeITER=0

distTransformTh=3

write=1


cv2.namedWindow('img')


#setup morphology kernels and FGBG params

kernel = np.ones((3,3), np.uint8)
hatkernel = np.ones((5,5), np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(history=fgbgHISTORY,detectShadows = 0 ,varThreshold = fgbgTHRESHOLD)
fgbgFAST = cv2.createBackgroundSubtractorMOG2(history=fgbgHISTORYfast,detectShadows = 0 ,varThreshold = 16)

### Prepare Image Function ###
### returns
###     crop_img (for display)
###     bodies (contours of bodies)
###     fullsegment (contours of combination of bodies, laplace edge, and motion; filtered for size and body intersection)
###     boxes (bounding boxes for segments)

def prepareImage(x):


    #eliminate background w/ dynamic bg sub"
    crop_img = x
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    im2, motion, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    fgmaskFAST=fgbgFAST.apply(gray)
    im2, motionFAST, hierarchy = cv2.findContours(fgmaskFAST,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #this eliminates the butts of the organisms so using an aggressive threshold
    #to bring back the butts and identify them as the homebase in the bodies list
    #creates a BODIESMASK to use to for making sure any contour touches a butt
    #demand reasonably circular butt
    #if butt is too big reject as well

    ret1,bodyth = cv2.threshold(gray,bodythresh,255,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(bodyth,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bodies=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if body_area_min < area < body_area_max:
            if 4*3.14*area/perimeter**2 >body_circ:
                bodies.append(cnt)
    bodiesmask = np.zeros(gray.shape, np.uint8)

    #distance transform to isolate overlappers and locate centroids from this
    cv2.drawContours(bodiesmask, bodies, -1, (255), thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(bodiesmask,cv2.DIST_L2,3)
    ret1,thr = cv2.threshold(dist_transform,distTransformTh,255,cv2.THRESH_BINARY)
    cent_th=np.uint8(thr)
    im2, centcontours, hierarchy = cv2.findContours(cent_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    centroids=[]
    for j in range(len(centcontours)):
        #grab the ith dist transformed contour
        bcnt=centcontours[j]

        if cv2.contourArea(bcnt)>1:
            M = cv2.moments(bcnt)
            #print("M values for"+str(j)+" are: "+str(M['m10'])+", "+str(M['m00'])+", "+str(M['m01'])+", "+str(M['m00']))
            cx = int(M['m10']/M['m00'])
            #print("Cx for "+str(j)+ "is "+str(cx))
            cy = int(M['m01']/M['m00'])
            #print("Cy for "+str(j)+ "is "+str(cy))
            centroids.append(np.array([cx,cy]))

    #combine motion adjusted with thresholded bodies
    full=cv2.add(fgmask,bodyth)

    #thin regions of neck are still often disconnected
    #use edge detection on tophat corrected+ dilation/erosion to connect these regions

    edges = cv2.Canny(gray,edgeMIN,edgeMAX)
    dilate = cv2.dilate(edges, kernel, iterations=dilateITER)
    full2=cv2.add(full,dilate)



    #locate contours in segmentedimage; filter for minimal area requirement and circularity;
    #create bounding rectangles for each contour
    #demand that contour touches a body segment

    im2, contours, hierarchy = cv2.findContours(full2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    fullsegments=[]
    boxes=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > body_area_min:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mask = np.zeros(full2.shape, np.uint8)
            cv2.drawContours(mask,[box],0,(255,0,255),thickness=cv2.FILLED)
            test=cv2.bitwise_and(mask,bodiesmask)
            if cv2.countNonZero(test) > minButtNeckIntersection:
                fullsegments.append(cnt)
                boxes.append(box)

    return crop_img,centroids,motion,motionFAST, fullsegments,bodies,boxes;


####

####
#### WRITE GRAYSCALE IMAGE AND ASSOCIATED CONTOURS TO 'DATA' FOLDER

def recordProcessed(crop,centroids,motion,motionFAST,full,bodies,count):
    impath=EXPFOLDER+'/data/imgs/'
    featurespath=EXPFOLDER+'/data/features/'
    #make image grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[framey1:framey2, framex1:framex2]
    cv2.imwrite(impath+str(count)+".jpg",gray)
    print("Wrote image")
    data=[centroids,motion,motionFAST,full,bodies]

    return;

### DISPLAY PROCESSING OF IMAGE ###
###     crop=img to be displayed
###     full=full contours
###     bodies=body contours
###     resize=scale for display
###

def displayProcessed(crop,centroids,motion,motionFAST,full,bodies,boxes,resize=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.drawContours(crop, full, -1, (255,255,255), thickness=3)
    cv2.drawContours(crop, motion, -1, (255,0,255), thickness=cv2.FILLED)
    cv2.drawContours(crop, motionFAST, -1, (0,255,255), thickness=cv2.FILLED)
    cv2.drawContours(crop, bodies, -1, (0,255,0), thickness=cv2.FILLED)
    for cent in centroids:
        bc=(cent[0],cent[1])
        cv2.putText(crop,'O',bc, font, 0,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(crop,'Frame # '+str(count),(25,25), font, 1,(255,255,255),2,cv2.LINE_AA)

    #follow along
    res = cv2.resize(crop_img,None,fx=resize, fy=resize, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img',res)

    return;

### END OF DISPLAY FUNCTIONS ###

### MAIN LOOP OF IMAGE ANALYSIS ####
segmentationData={}
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    origframe=frame
    if (type(frame) == type(None)):
        break
    #prepare contours from image
    crop_img,centroids,motion,motionFAST,fullcontours,bodycontours,boxes=prepareImage(frame)
    #cv2.imshow('img',crop_img)
    segmentationData[count]=[centroids,motion,motionFAST,fullcontours,bodycontours]



    if write==1:
        gray= cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(impath+str(count)+".jpg",gray)
        print("Wrote image "+str(count))
    if displayON==1:
        displayProcessed(crop_img,centroids,motion,motionFAST,fullcontours,bodycontours,boxes,resize=1.0)
    count=count+1
    print("Processed frame " +str(count))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
print("Pickling the data...")

fileObject=open(EXPFOLDER+'/data/pickledSegmentation',"wb")
pickle.dump(segmentationData,fileObject)
print("Done")
fileObject.close()


cv2.destroyAllWindows()
