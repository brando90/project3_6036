#this file is used to extract 1000*1000 matrix from Netflix Dataset
import numpy as np
import os
import operator as op
import sys
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
absPath=os.path.dirname(os.path.realpath(__file__))
relPath="/Netflix Dataset/training_set/training_set/"
path=absPath+relPath
dirList=os.listdir(path)
fileInfo=[]
numOfMovie=17763
thresOfMovie=1000
thresOfUser=1000
rewFreq=200

for i in xrange(numOfMovie):
    fileInfo+=[(path+dirList[i],os.path.getsize(path+dirList[i])/1000)];
fileInfo=sorted(fileInfo,key=op.itemgetter(1),reverse=True)
fileInfo=fileInfo[:thresOfMovie]
arr=np.ndarray([3,],dtype=np.int)
li=[]
for i in xrange(thresOfMovie):
    print i
    with open(fileInfo[i][0],"r") as f:
        idMovie=int(f.readline()[:-2])
        for line in f:  
            idUser, rating=line.split(",")[:2]
            idUser=int(idUser); rating=int(rating)
            li.append([idMovie,idUser,rating])
    if i==0:
        arr=np.array(li); li=[]; continue;
    if i%rewFreq==0:
        arr=np.vstack((arr,np.array(li))); li=[]
arr=np.vstack((arr,np.array(li))); li=[]
# play with the data
print "Select top "+str(thresOfUser)+" users"
tmp=itemfreq(arr[:,1])
tmp=tmp[tmp[:,1].argsort()[::-1]]
tmp=tmp[:thresOfUser,]
tmp=tmp[:,0][tmp[:,0].argsort()]
li=[]
arr=arr[arr[:,1].argsort()]
j=0
for i in xrange(len(arr)):
    if j<len(tmp) and arr[i,1]==tmp[j]:
        li.append(list(arr[i,]))
    elif j<len(tmp) and arr[i,1]>tmp[j]:
        j+=1
        if j<len(tmp) and arr[i,1]==tmp[j]:
            li.append(list(arr[i,]))
finalArr=np.array(li)
##mat=np.nan([thresOfUser,thresOfMovie])
##indX=0; indY=0
print "Relabel the users and movies"
j=0
for i in xrange(1,len(finalArr)):
    if finalArr[i,1]>finalArr[i-1,1]: finalArr[i-1,1]=j; j+=1
    else: finalArr[i-1,1]=j; 
finalArr[len(finalArr)-1,1]=j
finalArr=finalArr[finalArr[:,0].argsort()]
j=0
for i in xrange(1,len(finalArr)):
    if finalArr[i,0]>finalArr[i-1,0]: finalArr[i-1,0]=j; j+=1
    else: finalArr[i-1,0]=j; 
finalArr[len(finalArr)-1,0]=j
print "construct submatrix"
mat=np.zeros((thresOfUser,thresOfMovie),dtype=np.int)
for i in xrange(0,len(finalArr)):
    mat[finalArr[i,0],finalArr[i,1]]=finalArr[i,2]
##print "delete rows where 0 exist"
##while 1:
##    print mat.shape[0], mat.shape[1]
##    liRow=[]
##    for i in xrange(len(mat)):
##        liRow.append(np.count_nonzero(mat[i,]))
##    liCol=[]
##    for i in xrange(len(mat[0])):
##        liCol.append(np.count_nonzero(mat[:,i]))
##    if min(liRow)==mat.shape[1] and min(liCol)==mat.shape[0]:
##        break
##    else:
##        if min(liRow)<min(liCol):
##            mat=mat[np.logical_not(np.array(liRow)==min(liRow)),]
##        else:
##            mat=mat[:,np.logical_not(np.array(liCol)==min(liCol))]
            
        
