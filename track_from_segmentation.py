# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:16:47 2018

@author: scoyle
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import csv
import pickle


### PARAMETERS

EXPFOLDER="experiment"

framey1=5
framey2=1074
framex1=710
framex2=1800

width=framex2-framex1
height=framey2-framey1

font = cv2.FONT_HERSHEY_SIMPLEX

kernel = np.ones((3,3), np.uint8)

orgPersistTime=10*30 #window of frames to query whether an organism does or doesnt exist

frozenFlag=60 #how long before calling as frozen




### LOAD PICKLED DATA AND EXPERIMENTAL DATA FROM SEGMENTATION OUTPUT ####

def readPickle(filename):
    pickle_off=open(filename,"rb")
    data=pickle.load(pickle_off)
    return data;

def readData(experiment):
    dataset={}
    for file in os.listdir(experiment+'/data/features/'):
        fileData=readPickle(experiment+'/data/features/'+file)
        dataset[file]=fileData
        print("Read in file: "+str(file))
    return dataset;


### DIAGNOSTIC FUNCTIONS TO PLAYBACK EXTRACTED DATA ####

def playAllCentroids(data):
    timeordering=range(len(data.keys()))
    for timepoint in timeordering:
        blank=np.zeros((width,height), np.uint8)
        for cent in data[timepoint][0]:
            loc=(cent[0],cent[1])
            print("Location is "+str(loc))
            cv2.putText(blank,'IM HERE',loc, font, 0,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('img',blank)
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blank;

def playAllContours(data):
    timeordering=range(len(data.keys()))
    for timepoint in timeordering:
        blank=np.zeros((width,height), np.uint8)
        cont =data[timepoint][1]
        cv2.drawContours(blank, cont, -1, (255), thickness=cv2.FILLED)
        cv2.imshow('img',blank)
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blank;

def playAllBodies(data):
    timeordering=range(len(data.keys()))
    for timepoint in timeordering:
        blank=np.zeros((width,height), np.uint8)
        cont =data[timepoint][2]
        cv2.drawContours(blank, cont, -1, (255), thickness=cv2.FILLED)
        cv2.imshow('img',blank)
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blank;

def playData(data):
    timeordering=range(len(data.keys()))
    for timepoint in timeordering:
        blankG=np.zeros((width,height), np.uint8)
        blank = cv2.cvtColor(blankG, cv2.COLOR_GRAY2RGB)
        cont =data[timepoint][1]
        body =data[timepoint][2]
        cv2.drawContours(blank, cont, -1, (255,0,0), thickness=2)
        cv2.drawContours(blank, body, -1, (0,255,0), thickness=cv2.FILLED)
        for cent in data[timepoint][0]:
            loc=(cent[0],cent[1])
            print("Location is "+str(loc))
            cv2.putText(blank,'IM HERE',loc, font, 0,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('img',blank)
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return;

def showTimepoint(data,t,rsz=1,hat=0):
    img=cv2.imread('experiment/data/imgs/'+str(t)+'.jpg')
    if hat==1:
        img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    motion=data[t][1]
    motionFAST=data[t][2]
    cont =data[t][3]
    body =data[t][4]
    cv2.drawContours(img, cont, -1, (255,0,0), thickness=2)
    cv2.drawContours(img, motion, -1, (255,0,255), thickness=cv2.FILLED)
    cv2.drawContours(img, motionFAST, -1, (0,255,255), thickness=cv2.FILLED)
    cv2.drawContours(img, body, -1, (0,255,0), thickness=cv2.FILLED)
    for i in range(len(data[t][0])):
        cent=data[t][0][i]
        loc=(cent[0],cent[1])
        cv2.putText(img,str(i),loc, font, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(img,'IM HERE',loc, font, 0,(0,0,255),2,cv2.LINE_AA)
    res = cv2.resize(img,None,fx=rsz, fy=rsz, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('img',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img;



### FUNCTIONS THAT ARE USED AS PART OF TRACKING ###

def measureContourActivity(data,t,contourindex):
    #prepare blank
    img=cv2.imread('experiment/data/imgs/'+str(t)+'.jpg')
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvas=np.zeros_like(gray)

    #load data
    motion=data[t][1]
    motionFAST=data[t][2]
    cont =data[t][3][contourindex]
    body =data[t][4]

    #create imgs for intersection
    imgCont=np.zeros_like(gray)
    imgCont=cv2.fillPoly(imgCont, pts =[cont], color=(255,255,255))

    imgMotion=np.zeros_like(gray)
    imgMotion=cv2.drawContours(imgMotion, motion, -1, (255), thickness=cv2.FILLED)

    imgMotionFAST=np.zeros_like(gray)
    imgMotionFAST=cv2.drawContours(imgMotionFAST, motionFAST, -1, (255), thickness=cv2.FILLED)

    #compute intersections
    motionIntersection=np.count_nonzero(cv2.bitwise_and(imgCont,imgMotion))
    motionFASTIntersection=np.count_nonzero(cv2.bitwise_and(imgCont,imgMotionFAST))

    return motionIntersection,motionFASTIntersection;

def measureActivityInContours(data,t):
    contours=data[t][3]
    activities={}
    for i in range(len(contours)):
        mI,mFI=measureContourActivity(data,t,i)
        activities[i]=(mI,mFI)
    return activities;


def centroidNumberConflicts(data):
    centNum=[]
    for i in range(len(data.keys())):
        numcents=len(data[i][0])
        centNum.append(numcents)
    #check for changes; store difference in conflictsList
    conflictsList=[0]
    for i in range(len(centNum)):
        if i ==0:
            curCentNum=centNum[0]
        else:
            newCentNum=centNum[i]
            delta=newCentNum-curCentNum
            conflictsList.append(delta)
            curCentNum=newCentNum #update current value
            #print("We are on time "+str(i)+" and have "+str(curCentNum)+ "centroids")
    conflictsData=np.asarray(conflictsList)
    conflictsTimes=np.nonzero(conflictsData!=0)[0]
    return conflictsData,centNum;

def contoursNumberConflicts(data):
    contsNum=[]
    for i in range(len(data.keys())):
        numconts=len(data[i][3])
        contsNum.append(numconts)
    conflictsList=[0]
    for i in range(len(contsNum)):
        if i ==0:
            curContNum=contsNum[0]
        else:
            newContNum=contsNum[i]
            delta=newContNum-curContNum
            conflictsList.append(delta)
            curContNum=newContNum #update current value
    conflictsData=np.asarray(conflictsList)
    conflictsTimes=np.nonzero(conflictsData!=0)[0]
    return conflictsData,contsNum






###Initialize Tracking with data from timepoint t (0 timepoint usually has issues)

def initializeTracking(data,t=1):

    #extract values
    centroids=data[t][0]
    motion=data[t][1]
    motionFAST=data[t][2]
    conts =data[t][3]
    bodies =data[t][4]

    #create blank list or organisms
    organismsList={}

    for j in range(len(centroids)):
        #initilaize organism dictionary
        organism={}
        #createblankdata
        data=[]

        #timestamp is the key for the dictionary
        timepoint=t

        ##first initialize the organisms by centroids
        numpCent=centroids[j]
        data.append(numpCent)

        #convert np to tuple for some stuff
        bodycenter=(numpCent[0],numpCent[1])

        #check each fullcontour for whether it maps to the bodycenter
        fullconts=[]
        for i in range(len(conts)):
            TestContour=cv2.pointPolygonTest(conts[i],bodycenter,measureDist=False)
            fullconts.append(TestContour)

        index=np.argmax(fullconts)
        data.append(conts[index])

        #check each body contour for whtehr it maps to the bodycenter
        bodyconts=[]
        for i in range(len(bodies)):
            TestContour=cv2.pointPolygonTest(bodies[i],bodycenter,measureDist=False)
            bodyconts.append(TestContour)

        index=np.argmax(bodyconts)
        data.append(bodies[index])

        CONFLICT_FLAG='false'
        data.append(CONFLICT_FLAG)
        ## at this point data is a list that contains (body center, full contour, body contour) as its entries

        ## now associate this data with the timepoint

        organism[timepoint]=data

        ## now add this organism to the list of all orgnaisms from the experiment

        organismsList[j]=organism
        #print("Stored organism "+str(j) + "in dictionary!")

    # create active organisms SET that contains all the organisms
    activelyTracking=set(range(len(organismsList)))

    ## DIAGNOSTIC TO CHECK THAT ORGANISM ID AND ITS CONTOUR LINE UP IN PHYSICAL SPACE)
    img=cv2.imread('experiment/data/imgs/'+str(t)+'.jpg')
    mask=np.zeros_like(img)


    for i in activelyTracking:
        cv2.fillPoly(mask, pts =[organismsList[i][timepoint][1]], color=(100+i*10,150+i*5,100+i*2.5))
        #print("Printed contour for organism "+str(i)+" at timepoint " +str(timepoint))
        cv2.fillPoly(mask, pts =[organismsList[i][timepoint][2]], color=(0,50+i*10,0))
        #print("Printed body for organism "+str(i)+" at timepoint " +str(timepoint))
        cv2.putText(mask,'Org '+str(i),(organismsList[i][timepoint][0][0],organismsList[i][timepoint][0][1]), font, 1,(0,150+i*5,100+i*5),2,cv2.LINE_AA)
        cv2.putText(mask,'Org '+str(i),(organismsList[i][timepoint][0][0],organismsList[i][timepoint][0][1]), font, 0,(0,0,255),2,cv2.LINE_AA)
        #print("Printed Label for organism "+str(i)+" at timepoint " +str(timepoint))
        cv2.putText(mask,'Frame # '+str(timepoint),(25,25), font, 1,(255,255,255),2,cv2.LINE_AA)
    res = cv2.resize(mask,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    frozens={}
    for o in activelyTracking:
        frozens[o]=0
    ###seed the conflicts data for contours and centroids

    return organismsList,activelyTracking,frozens,res;

def updateTracking(organismsList,data,t,activelyTracking,centConflictsData,contConflictsData,centsAtTime,contsAtTime,frozens):
    img=cv2.imread('experiment/data/imgs/'+str(t)+'.jpg')
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #extract values
    #current:
    centroids=data[t][0]
    motion=data[t][1]
    motionFAST=data[t][2]
    conts =data[t][3]
    bodies =data[t][4]

    #prev:
    centroidsPrev=data[t-1][0]
    motionPrev=data[t-1][1]
    motionFASTPrev=data[t-1][2]
    contsPrev =data[t-1][3]
    bodiesPrev =data[t-1][4]

    curNumActive=len(activelyTracking)
    delta=len(centroids)-curNumActive


    #now see if we have a contour conflict
    #since organisms aren't defined by contour only need to resolve contour mergers

    ###first align new centroids to their most probable bodies and contours

    currentTimepointData=[]
    for j in range(len(centroids)):
        #initilaize  data associated with centroid
        data=[]

        ## first deposit the centroid in the data
        numpCent=centroids[j]
        data.append(numpCent)

        #convert np to tuple for some stuff
        bodycenter=(numpCent[0],numpCent[1])

        #check each fullcontour for whtehr it maps to the bodycenter
        fullconts=[]
        for i in range(len(conts)):
            TestContour=cv2.pointPolygonTest(conts[i],bodycenter,measureDist=False)
            fullconts.append(TestContour)

        index=np.argmax(fullconts)
        data.append(conts[index])

        #check each body contour for whtehr it maps to the bodycenter
        bodyconts=[]
        for i in range(len(bodies)):
            TestContour=cv2.pointPolygonTest(bodies[i],bodycenter,measureDist=False)
            bodyconts.append(TestContour)

        index=np.argmax(bodyconts)
        data.append(bodies[index])

        CONFLICT_FLAG='false'
        data.append(CONFLICT_FLAG)
        #now add this centroids data to the dataset
        currentTimepointData.append(data)

    #######
    ####### now with this new data, compare it to previous frame to align it
    #######

    # first extract centroids from newdata in order
    newcents=[]
    for i in range(len(currentTimepointData)):
        centroid=currentTimepointData[i][0]
        newcents.append(centroid)
    newcents=np.asarray(newcents)

    ## Now compare previous frames data of active organisms to current frames new list of data
    ## and add new timepoint data to matching organisms from Active Organisms list

    usedOrganisms=set()
    for AOindex in activelyTracking:

        #get previousCentroid
        prevCentroid=organismsList[AOindex][t-1][0]
        prevCenter=(prevCentroid[0],prevCentroid[1])

        ##now identify the appropriate organism to attach this data to by
        ##check which previous centroid is closest to each current centroid

        dist_2 = np.sum((newcents - prevCenter)**2, axis=1)
        listIndex=np.argmin(dist_2) #which organism are we?

        #print("I mapped to new data" +str(listIndex))
        ## now update that organisms dictionary to map the current timestamp to the current data
        organismsList[AOindex][t]=currentTimepointData[listIndex]

        usedOrganisms.add(listIndex) #contains indices from newCents that were used up.

    print("The used organisms are...")
    print(usedOrganisms)

    unusedOrganisms=set(range(len(newcents))).difference(usedOrganisms)
    print("the unused organisms are...")
    print(unusedOrganisms)

     ###now check out the full contours for duplicates as well
    usedInTest=set()
    for org in activelyTracking:
        if org in usedInTest:
            print("Already tested "+str(org))
        else:
            for testOrg in activelyTracking:
                if org!=testOrg:   #dont check yourself

                    '''fastest'''
                    orgBody=organismsList[org][t][1]
                    testOrgBody=organismsList[testOrg][t][1]
                    if np.array_equal(orgBody,testOrgBody)==True:
                        print("Org "+str(org)+" and "+str(testOrg)+" have the same contour oh noes!")

                        ##fix similar to above but only do operations on Full contour [1]
                        orgPrev=organismsList[org][t-1]

                        orgCent=orgPrev[0]
                        orgFull=orgPrev[1]
                        orgBody=orgPrev[2]

                        testOrgPrev=organismsList[testOrg][t-1]

                        testOrgCent=testOrgPrev[0]
                        testOrgFull=testOrgPrev[1]
                        testOrgBody=testOrgPrev[2]


                        ###FIRST: (org)

                        curFrame=np.zeros_like(gray)
                        cv2.fillPoly(curFrame,pts=[organismsList[org][t][1]], color=(255))


                        orgFrame=np.zeros_like(gray)
                        cv2.fillPoly(orgFrame, pts =[orgFull], color=(255))

                        testFrame=np.zeros_like(gray)
                        cv2.fillPoly(testFrame, pts =[testOrgFull], color=(255))

                        orgFrameANDcurFrame=cv2.bitwise_and(curFrame,cv2.bitwise_not(testFrame))
                        orgErode = cv2.erode(orgFrameANDcurFrame, kernel, iterations=2)
                        orgDilate = cv2.dilate(orgErode, kernel, iterations=2)


                        im2, orgContours, hierarchy = cv2.findContours(orgDilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        areas = [cv2.contourArea(c) for c in orgContours]

                        if len(areas)==0:
                            orgContour=orgBody #freeze to body contour
                            #print("danger had to freeze contour for "+str(org))
                            val=frozens[org]+1
                            frozens[org]=val
                            #print("CONT Changing frozen for "+str(org)+" to "+str(val))
                        elif max(areas)==0: #no contour detected?
                            orgContour=orgBody #freeze to body contour
                            #print("danger had to freeze contour for "+str(org))
                            val=frozens[org]+1
                            frozens[org]=val
                            #print("CONT changing frozen for "+str(org)+" to "+str(val))
                        else:
                            max_index = np.argmax(areas)
                            orgContour=orgContours[max_index]
                            frozens[org]=0
                            #print("CONT reset frozen for "+str(org)+ "to 0!")


                        #####SECOND (testOrg)

                        curFrame=np.zeros_like(gray)
                        cv2.fillPoly(curFrame,pts=[organismsList[testOrg][t][1]], color=(255))

                        orgFrame=np.zeros_like(gray)
                        cv2.fillPoly(orgFrame, pts =[orgFull], color=(255))

                        testFrame=np.zeros_like(gray)
                        cv2.fillPoly(testFrame, pts =[testOrgFull], color=(255))

                        testFrameANDcurFrame=cv2.bitwise_and(curFrame,cv2.bitwise_not(orgFrame))
                        testErode = cv2.erode(testFrameANDcurFrame, kernel, iterations=2)
                        testDilate = cv2.dilate(testErode, kernel, iterations=2)

                        im2, testContours, hierarchy = cv2.findContours(testDilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        areas = [cv2.contourArea(c) for c in testContours]

                        if len(areas)==0:
                            testContour=testOrgBody #freeze prefvious contour in that case
                            #print("danger had to freeze contour for"+str(testOrg))
                            val=frozens[testOrg]+1
                           # print("CONT Changing frozen for "+str(testOrg)+ "to "+str(val))
                            frozens[testOrg]=val
                        elif max(areas)==0: #no contour detected?
                            testContour=testOrgFull #freeze prefvious contour in that case
                            #print("danger had to freeze contour for"+str(testOrg))
                            val=frozens[testOrg]+1
                            #print("CONT Changing frozen for "+str(testOrg)+ "to "+str(val))
                            frozens[testOrg]=val
                        else:
                            max_index = np.argmax(areas)
                            testContour=testContours[max_index]
                            frozens[testOrg]=0
                            #print("CONT reset frozen for "+str(testOrg)+" to 0!")
                        #mask2=np.zeros_like(gray)
                        #cv2.fillPoly(mask2, pts =[contour16], color=(155))



                        ##now update the organismslist

                        newOrgCent=orgCent
                        newOrgFull=orgContour #keeping contour for now
                        newOrgBody=orgBody

                        newOrg=[newOrgCent,newOrgFull,newOrgBody,'true'] #conflict flag set to TRUE for this frame

                        organismsList[org][t]=newOrg

                        newTestOrgCent=testOrgCent
                        newTestOrgFull=testContour
                        newTestOrgBody=testOrgBody

                        newTestOrg=[newTestOrgCent,newTestOrgFull,newTestOrgBody,'true'] #conflict flag set to TRUE for this frame

                        organismsList[testOrg][t]=newTestOrg
                        #print("I repaired the organisms!")

    ###we need to split any organisms that were assigned the same bodycontour
    ### locate any matches
    usedInTest=set()

    for org in activelyTracking:
        if org in usedInTest:
            print("Already tested "+str(org))
        else:
            for testOrg in activelyTracking:
                if org!=testOrg:   #dont check yourself

                    '''fastest'''
                    orgBody=organismsList[org][t][2]
                    testOrgBody=organismsList[testOrg][t][2]
                    if np.array_equal(orgBody,testOrgBody)==True:
                        print("Org "+str(org)+" and "+str(testOrg)+" are the same!")
                        usedInTest.add(testOrg)
                        #in event two organism are the same, just freeze everything until they
                        #aren't the same anymore

                        ##test if can unpack and repack
                        orgPrev=organismsList[org][t-1]

                        orgCent=orgPrev[0]
                        orgFull=orgPrev[1]
                        orgBody=orgPrev[2]

                        testOrgPrev=organismsList[testOrg][t-1]

                        testOrgCent=testOrgPrev[0]
                        testOrgFull=testOrgPrev[1]
                        testOrgBody=testOrgPrev[2]


                        ###FIRST: (org)

                        curFrame=np.zeros_like(gray)
                        cv2.fillPoly(curFrame,pts=[organismsList[org][t][2]], color=(255))


                        orgFrame=np.zeros_like(gray)
                        cv2.fillPoly(orgFrame, pts =[orgBody], color=(255))

                        testFrame=np.zeros_like(gray)
                        cv2.fillPoly(testFrame, pts =[testOrgBody], color=(255))

                        orgFrameANDcurFrame=cv2.bitwise_and(curFrame,cv2.bitwise_not(testFrame))

                        im2, orgContours, hierarchy = cv2.findContours(orgFrameANDcurFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        areas = [cv2.contourArea(c) for c in orgContours]

                        if len(areas)==0: #lost it
                            orgContour=orgBody
                            orgCx=orgCent[0]
                            orgCy=orgCent[1]
                            val=frozens[org]+1
                            frozens[org]=val
                            #print("CENT Changing frozen for "+str(org)+" to "+str(val))
                        elif max(areas)==0: #lost it
                            orgContour=orgBody
                            orgCx=orgCent[0]
                            orgCy=orgCent[1]
                            val=frozens[org]+1
                            frozens[org]=val
                            #print("CENT Changing frozen for "+str(org)+" to "+str(val))
                        else:
                            max_index = np.argmax(areas)
                            orgContour=orgContours[max_index]
                            orgM = cv2.moments(orgContour)
                            orgCx = int(orgM['m10']/orgM['m00'])
                            orgCy = int(orgM['m01']/orgM['m00'])
                            #print("New centroid for"+str(org)+" is "+str(orgCx)+","+str(orgCy))
                            frozens[org]=0
                            #print("CENT reset frozen for "+str(org)+" to 0!")
                        #mask=np.zeros_like(gray)
                        #cv2.fillPoly(mask, pts =[contour15], color=(155))




                        #####SECOND (testOrg)

                        curFrame=np.zeros_like(gray)
                        cv2.fillPoly(curFrame,pts=[organismsList[testOrg][t][2]], color=(255))

                        orgFrame=np.zeros_like(gray)
                        cv2.fillPoly(orgFrame, pts =[orgBody], color=(255))

                        testFrame=np.zeros_like(gray)
                        cv2.fillPoly(testFrame, pts =[testOrgBody], color=(255))

                        testFrameANDcurFrame=cv2.bitwise_and(curFrame,cv2.bitwise_not(orgFrame))

                        im2, testContours, hierarchy = cv2.findContours(testFrameANDcurFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                        areas = [cv2.contourArea(c) for c in testContours]
                        if len(areas)==0:
                            testContour=testOrgBody #freeze
                            testOrgCx = testOrgCent[0]
                            testOrgCy = testOrgCent[1]
                            val=frozens[testOrg]+1
                            frozens[testOrg]=val
                            #print("CENT Changing frozen for "+str(testOrg)+ "to "+str(val))
                        elif max(areas)==0:
                            testContour=testOrgBody #freeze
                            testOrgCx = testOrgCent[0]
                            testOrgCy = testOrgCent[1]
                            val=frozens[testOrg]+1
                            frozens[testOrg]=val
                            #print("CENT Changing frozen for "+str(testOrg)+ "to "+str(val))
                        else:
                            max_index = np.argmax(areas)
                            testContour=testContours[max_index]
                            testOrgM = cv2.moments(testContour)
                            testOrgCx = int(testOrgM['m10']/testOrgM['m00'])
                            testOrgCy = int(testOrgM['m01']/testOrgM['m00'])
                            #print("New centroid for 16 is "+str(testOrgCx)+","+str(testOrgCy))
                            frozens[testOrg]=0
                            #print("CENT reset frozen for"+str(testOrg)+" to 0!")
                        #mask2=np.zeros_like(gray)
                        #cv2.fillPoly(mask2, pts =[contour16], color=(155))




                        ##now update the organismslist

                        newOrgCent=np.asarray([orgCx,orgCy])
                        newOrgFull=organismsList[org][t][1] #keeping contour for now
                        newOrgBody=orgContour

                        newOrg=[newOrgCent,newOrgFull,newOrgBody,'true'] #conflict flag set to TRUE for this frame

                        organismsList[org][t]=newOrg

                        newTestOrgCent=np.asarray([testOrgCx,testOrgCy])
                        newTestOrgFull=organismsList[testOrg][t][1]
                        newTestOrgBody=testContour

                        newTestOrg=[newTestOrgCent,newTestOrgFull,newTestOrgBody,'true']

                        organismsList[testOrg][t]=newTestOrg
                        print("I repaired the organisms!")





    ###add any new organisms

    if len(usedOrganisms)<np.median(contsAtTime[t:(t+orgPersistTime)]):
        orgID=len(organismsList.keys())
        for org in unusedOrganisms:
                newOrg={}
                newOrg[t]=currentTimepointData[org]
                organismsList[orgID]=newOrg
                activelyTracking.add(orgID)
                frozens[orgID]=0
                print("Created new organism, "+str(orgID))
                orgID=orgID+1

    ###remove from activelyTracking if frozen > frozenFlag times

    removalSet=set()
    for org in activelyTracking:
        if frozens[org]>frozenFlag:
            removalSet.add(org)
            print("Had to remove "+str(org)+ " for being frozen too long!")

    new=activelyTracking.difference(removalSet)
    activelyTracking=new


    ## DIAGNOSTIC TO CHECK THAT ORGANISM ID AND ITS CONTOUR LINE UP IN PHYSICAL SPACE)

    for i in activelyTracking:
        cv2.polylines(img, pts =[organismsList[i][t][1]],isClosed=True, thickness=2, color=(255,0,0))
        #print("Printed contour for organism "+str(i)+" at timepoint " +timepoint)
        cv2.polylines(img, pts =[organismsList[i][t][2]], isClosed=True, thickness=2,color=(0,255,0))
        #print("Printed body for organism "+str(i)+" at timepoint " +timepoint)
    for i in activelyTracking:
        cv2.putText(img,str(i),(organismsList[i][t][0][0],organismsList[i][t][0][1]), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'Org '+str(i),(organismsList[i][t][0][0],organismsList[i][t][0][1]), font, 0,(0,0,255),2,cv2.LINE_AA)
        #print("Printed Label for organism "+str(i)+" at timepoint " +timepoint)
        cv2.putText(img,'Frame # '+str(t),(25,25), font, 1,(255,255,255),2,cv2.LINE_AA)
    res = cv2.resize(img,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)

    print("Finished track of time "+str(t))
    return organismsList,activelyTracking,frozens,res;


#### EXAMPLE SCRIPT BELOW FOR PROCESSING DATA OUTPUT FROM SEGMENTATION SCRIPT 

if __name__ == "__main__":
    mydata=readPickle('experiment/data/pickledSegmentation')
    centConf,centsAtTime=centroidNumberConflicts(mydata)
    contConf,contsAtTime=contoursNumberConflicts(mydata)
    
    start=1
    finish=100
    for i in range(start,finish):
        if i==start:
            oList,aT,frozens,res=initializeTracking(mydata,t=i)
            print("Initialized at "+str(i))
            #cv2.imshow('img',res)
        if i>start:
            oList,aT,frozens,res=updateTracking(oList,mydata,i,aT,centConf,contConf,centsAtTime,contsAtTime,frozens)
            #print("Updated at "+str(i))
            #cv2.imshow('img',res)
        cv2.imshow('img',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Pickling Organisms Tracks...")
    fileObject=open(EXPFOLDER+'/tracks/pickledTracks',"wb")
    pickle.dump(oList,fileObject)
    print("Done")
    fileObject.close()
    print("finished!")
