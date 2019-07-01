import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import csv
import pickle
import glob
import errno
import scipy

#from seaborn import *
from scipy.cluster.hierarchy import dendrogram, linkage
from numpy import genfromtxt
from numpy import *
from pylab import *
from scipy import signal
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from matplotlib import animation
from matplotlib.mlab import PCA
from scipy import signal


from scipy.interpolate import interp2d

from skimage.morphology import skeletonize
from skimage.morphology import medial_axis

#import statsmodels.api as sm
#lowess = sm.nonparametric.lowess

import random

imgH=1069
imgW=1090

kernel=np.asarray([[	0,	0,	0,	0,	1,	0,	0,	0,	0	],[	0,	0,	0,	1,	1,	1,	0,	0,	0	],[	0,	0,	1,	1,	1,	1,	1,	0,	0	],[	0,	1,	1,	1,	1,	1,	1,	1,	0	],[	1,	1,	1,	1,	1,	1,	1,	1,	1	],[	0,	1,	1,	1,	1,	1,	1,	1,	0	],[	0,	0,	1,	1,	1,	1,	1,	0,	0	],[	0,	0,	0,	1,	1,	1,	0,	0,	0	],[	0,	0,	0,	0,	1,	0,	0,	0,	0	]])


##########################################################
##### READING AND WRITING DATA/EXPERIMENT FUNCTIONS ######

def readPickle(filename):
    print("reading "+str(filename))
    pickle_off=open(filename,"rb")
    data=pickle.load(pickle_off)
    return data;

'''
THE DATA IN A TRACK IS STORED AS A DICTIONARY
WHERE FOR A TIMEPOINT T, ORG[T] RETURNS A VECTOR WITH THE 
FOLLOWING CONTENTS IN THE INDICATED INDEX

    0: centroid
    1: full contour
    2: body contour
    3: conflict flag (from tracking; says if organisms collided)
    4: skeleton points
    5: skeleton head
    6: skeleton tail
    7: oriented neck points (from skeleton); goes from base of neck to tip of neck
    8: oriented body points (from skeleton); goes from tail to start of neck
    9: contains a 3-uple in which [0] is the centerline, [1] is the x points, and [2] is the y-points
    10: theta representation of shape
    11: necklength
    12: eigenNecks matrix
    13: covariance explained
    14: 5-ev fit of shape

'''

### THESE DATA CAN BE PLOTTED AND MANIPULATED IN ANY WAY USING STANDARD MATPLOTLIB
### PLOTS OR OTHER PYTHON PACKAGES

### FOR EXAMPLE, THE FOLLOWING FUNCTIONS CAN BE USED TO GENERATE A REACH PLOT FOR
### AN ORGANISM FOR A START AND ENDING TIMEPOINT

def bodyOrientedAngleFromFit(orgT):
    bx=orgT[8].T[0][0]
    by=imgH-orgT[8].T[1][0]
    ranX=max(bx)-min(bx)
    ranY=max(by)-min(by)
    if ranX>ranY:
        fit=np.polyfit(bx, by, 1)
        print("Fitting along X-axis")
        bx0=orgT[8].T[0][0][0]
        bxf=orgT[8].T[0][0][-1]
        fy0=bx0*fit[0]+fit[1]
        fyf=bxf*fit[0]+fit[1]
        dx=bxf-bx0
        dy=fyf-fy0
        angle=np.arctan2(dy,dx)
        print(angle)
    else:
        fit=np.polyfit(by,bx,1)
        print("Fitting along Y-axis; more points to work with there")
        by0=orgT[8].T[1][0][0]
        byf=orgT[8].T[1][0][-1]
        fx0=by0*fit[0]+fit[1]
        fxf=byf*fit[0]+fit[1]
        dx=fxf-fx0
        dy=byf-by0
        angle=np.arctan2(dy,dx)+np.pi
        print(angle)
    return angle;

def rotateAndCenterHeadByBodyAngle(orgT):
    nx=orgT[7].T[0][0]
    ny=imgH-orgT[7].T[1][0]
    angle=bodyOrientedAngleFromFit(orgT)
    
    #plt.plot(orgT[1].T[0][0],imgH-orgT[1].T[1][0],'ro')
    #plt.plot(nx,ny,'bo')
    
    #translate base of neck to 0,0:
    transNx=[]
    transNy=[]
    for i in range(len(nx)):    
        transNx.append(nx[i]-nx[0])
        transNy.append(ny[i]-ny[0])
    #plt.plot(transNx,transNy,'ro')
    rotNx=[]
    rotNy=[]
    for i in range(len(transNx)):
        x=transNx[i]
        y=transNy[i]
        rho=np.sqrt(x**2+y**2)
        phi=np.arctan2(y,x)
        #rotate phi by theta
        newPhi=phi-angle
        #convert back to cartesian
        rotNx.append(rho*np.cos(newPhi))
        rotNy.append(rho*np.sin(newPhi))
    #plt.plot(rotNx[-2:],rotNy[-2:],'go')
    #rotate
    return rotNx,rotNy;


def countAllHeadPositionsReach(org,rmin,rmax):
    matrix=np.zeros((1000,1000))
    lengths=[]
    for t in range(rmin,rmax):
        try:
            rx,ry=rotateAndCenterHeadByBodyAngle(org[t])
            #print(rx)
            #print(ry)
            headX=int(rx[-1])+500
            #print(headX)
            headY=int(ry[-1])+500
            #print(headY)
            
            
            matrix[headY][headX]=matrix[headY][headX]+1
            '''

            matrix[headY+1][headX]=matrix[headY+1][headX]+1
            matrix[headY-1][headX]=matrix[headY-1][headX]+1
            matrix[headY][headX+1]=matrix[headY][headX+1]+1
            matrix[headY][headX-1]=matrix[headY][headX-1]+1
            matrix[headY+1][headX+1]=matrix[headY+1][headX+1]+1
            matrix[headY-1][headX-1]=matrix[headY-1][headX-1]+1
            matrix[headY+1][headX-1]=matrix[headY+1][headX-1]+1
            matrix[headY-1][headX+1]=matrix[headY-1][headX+1]+1
            '''
            print("did "+str(t))
            if not org[t][11]:
                print("skipping")
            else:
                lengths.append(org[t][11])
        except:
            print("Bad fit precluded this datapoint.")
    matOnes=np.ones_like(matrix)
    print(matrix.shape)
    print(matOnes.shape)
    nonZero=(matOnes*matrix!=0)
    nonZero=nonZero*1
    conv=scipy.signal.convolve2d(nonZero,kernel)
    reachSize=np.count_nonzero(conv)
    #reachSize=np.count_nonzero(matrix)
    meanL=np.mean(lengths)
    sd=np.std(lengths)
    maxL=np.percentile(lengths,99)
    #maxL=meanL+2*sd
    print(reachSize)
    print(meanL)
    #plt.matshow(matrix,cmap='jet',vmin=0, vmax=50)
    #plt.colorbar()
    data=[meanL,maxL,reachSize,matrix,conv]
    return data;

### RUN THESE FUNCTIONS TO PRODUCE A REACH PLOT ON
### ORGANISM 11 FRAMES 20 THROUGH 16000

if __name__ == "__main__":
    org146=readPickle("data/146")
    data=countAllHeadPositionsReach(org146,114051,139300)
    plt.imshow(data[4])