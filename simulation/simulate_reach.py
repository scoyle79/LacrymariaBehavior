import seaborn as sns
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline



## Load MyD, e1shape, es2shape, e3shape, e4shape into data before running simulation.

e1shape=np.load('e1shape.npy')
e2shape=np.load('e2shape.npy')
e3shape=np.load('e3shape.npy')
e4shape=np.load('e4shape.npy')

myD=np.load('myD.npy')


## functions for simulation


def thetaToShape(theta,ds=1):
    xs=[]
    ys=[]
    for i in range(len(theta)):
        if i==0:
            xs.append(0)
            ys.append(0)
        else:
            xs.append(xs[i-1]+ds*np.cos(theta[i]))
            ys.append(ys[i-1]+ds*np.sin(theta[i]))
    shape=np.asarray([xs,ys])
    return shape;

def reconstructReachFromShape(l=60,myE1=0,myE2=0,myE3=0,myE4=0,ext=20):
    lMax=l+ext
    shape=e1shape*myE1+e2shape*myE2+e3shape*myE3+e4shape*myE4
    real=thetaToShape(shape,ds=l/98)
    tipAngle=np.arctan((real[1][-1]-real[1][-2])/(real[0][-1]-real[0][-2]))
    endToEndDist=np.sqrt(real[0][-1]**2+real[1][-1]**2)
    print(endToEndDist)
    #plt.plot(real[0],real[1])
    startX=real[0][-1]
    startY=real[1][-1]
    prevX=real[0][-2]
    prevY=real[1][-2]
    
    xpoints=[startX]
    ypoints=[startY]
    if startX>prevX: #pointing forward:
        endDistance=startX
        while endDistance<lMax:
            extX=startX+np.cos(tipAngle)
            extY=startY+np.sin(tipAngle)
            xpoints.append(extX)
            ypoints.append(extY)
            endDistance=np.sqrt(extX**2+extY**2)
            startX=extX
            startY=extY
            #print(endDistance)
    else:
        endDistance=startX
        while endDistance<lMax:
            extX=startX-np.cos(tipAngle)
            extY=startY-np.sin(tipAngle)
            xpoints.append(extX)
            ypoints.append(extY)
            endDistance=np.sqrt(extX**2+extY**2)
            startX=extX
            startY=extY
            #print(endDistance)
    return xpoints,ypoints;

def simulateShapesWithTimeVaryingL0(lfunction,steps=20000):
    xlist=[]
    ylist=[]
    los=[]
    ls=[]
    for t in range(steps):
        try:
            myL=lfunction(t)
            los.append(myL)
            sim1=np.random.exponential(scale=myD[1][myD[0]==myL])
            sim2=np.random.exponential(scale=myD[2][myD[0]==myL])
            sim3=np.random.exponential(scale=myD[3][myD[0]==myL])
            sim4=np.random.exponential(scale=myD[4][myD[0]==myL])
            myExt=np.random.exponential(scale=extensileFunction(myL/2))/2
            ls.append(myL+myExt)
            xs,ys=reconstructReachFromShape(l=myL,myE1=sim1,myE2=sim2,myE3=sim3,myE4=sim4,ext=myExt)
            xlist=xlist+xs
            ylist=ylist+ys
            xs,ys=reconstructReachFromShape(l=myL,myE1=-sim1,myE2=-sim2,myE3=-sim3,myE4=-sim4,ext=myExt)
            xlist=xlist+xs
            ylist=ylist+ys
            print(t)
        except:
            print("could not simulate at time step " +str(t))
    fig,axs=plt.subplots(3)
    axs[0].plot(xlist,ylist,'o',alpha=0.01)
    axs[1].plot(los)
    axs[1].plot(ls)
    axs[2].hist2d(xlist,ylist,normed=True,bins=100,cmax=0.0001)
    return xlist,ylist;

def simulateShapesWithTimeVaryingL0_noShape(lfunction,steps=20000):
    xlist=[]
    ylist=[]
    los=[]
    ls=[]
    for t in range(steps):
        try:
            myL=lfunction(t)
            los.append(myL)
            sim1=0
            sim2=0
            sim3=0
            sim4=0
            myExt=0
            ls.append(myL+myExt)
            xs,ys=reconstructReachFromShape(l=myL,myE1=sim1,myE2=sim2,myE3=sim3,myE4=sim4,ext=myExt)
            xlist=xlist+xs
            ylist=ylist+ys
            xs,ys=reconstructReachFromShape(l=myL,myE1=-sim1,myE2=-sim2,myE3=-sim3,myE4=-sim4,ext=myExt)
            xlist=xlist+xs
            ylist=ylist+ys
            print(t)
        except:
            print("could not simulate at time step " +str(t))
    fig,axs=plt.subplots(3)
    axs[0].plot(xlist,ylist,'o',alpha=0.01)
    axs[1].plot(los)
    axs[1].plot(ls)
    axs[2].hist2d(xlist,ylist,normed=True,bins=100,cmax=0.0001)
    return xlist,ylist;


def extensileFunction(L0):
    return L0*.21711657017973526+10.488598113822984;

def contractileFunction(L0):
    return np.abs(L0*(-0.26804208859686174)-6.079376962446371);

def sinL(t,rmin=25,rmax=99,per=1/5000):
    return int((rmax+rmin)/2+(rmax-rmin)*np.sin(per*t-1000)/2);



### EXAMPLE SIMULATION USING FUNCTIONS:
    
if __name__ == "__main__":
    mySimX,mySimY=simulateShapesWithTimeVaryingL0(sinL,steps=30000)
    
    ##mySimx and mySimY contain the x and y positions of the head derived from the simulation