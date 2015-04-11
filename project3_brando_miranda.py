import random as ra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA

# Returns the full toy data matrix.
# Output: X: Toy data matrix.
def readToyData():
    X=np.zeros((500,2),dtype=float)
    i=0
    with open("toy_data.txt","r") as f:
        for line in f:
            X[i,]=map(float,line.split(" "))
            i+=1
    return X   
# Returns the netlix data matrix.
# Output: X: netflix data matrix.
def readNetflixData():
    X=np.zeros((1000,1000),dtype=float)
    i=0
    with open("1000mat.txt","r") as f:
        for line in f:
            X[i,]=map(float,line.split(" "))
            i+=1
    return X 
# plot 2D toy data
# input: X: n*d data matrix; K: number of mixtures; Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        Label: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        title: a string represents the title for the plot
def plot2D(X,K,Mu,P,Var,Label,title):
    r=0.5
    color=["r","g","y","b","k","m","c"]
    d=len(Label[0])
    per=np.divide(Label*1.0,np.tile(np.sum(Label,axis=1,keepdims=True),(1,d)))
    fig=plt.figure()
    plt.title(title)
    ax=plt.gcf().gca()
    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))
    #print per
    for i in xrange(len(X)):
        angle=0
        for j in xrange(K):
            cir=pat.Arc((X[i,0],X[i,1]),r,r,0,angle,angle+per[i,j]*360,edgecolor=color[j])
            ax.add_patch(cir)
            angle+=per[i,j]*360
    for j in xrange(K):
        circle=plt.Circle((Mu[j,0],Mu[j,1]),Var[j]**0.5,color=color[j],fill=False)
        ax.add_artist(circle)
        text=plt.text(Mu[j,0],Mu[j,1],"mu=("+str("%.2f" %Mu[j,0])+","+str("%.2f" %Mu[j,1])+"),sigma="+str("%.2f" %Var[j]))
        ax.add_artist(text)
    plt.show()
def normalDensity(x,K,Mu,Var):
    num=np.sum(np.square(np.tile(x,(K,1))-Mu),axis=1,keepdims=True)
    den=2*Var
    norDen=np.divide(np.exp(-np.divide(num*1.0,den))*1.0,np.sqrt(2*np.pi*Var))
    return (norDen)    
# K Means method
# input: X: n*d data matrix; K: number of mixtures; Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def kMeans(X, K, Mu, P, Var):
    prevCost=-1.0; curCost=0.0
    n=len(X)
    d=len(X[0])
    while abs(prevCost-curCost)>1e-4:
        post=np.zeros((n,K))
        prevCost=curCost
        #E step
        for i in xrange(n):
            post[i,np.argmin(np.sum(np.square(np.tile(X[i,],(K,1))-Mu),axis=1))]=1
        #M step
        n_hat=np.transpose(np.sum(post,axis=0,keepdims=True))
        P=n_hat*1.0/n
        for i in xrange(K):
            Mu[i,]=np.sum(np.multiply(X,np.tile(post[:,i:i+1],(1,d))),axis=0,keepdims=True)*1.0/n_hat[i]
            Var[i]=np.dot(post[:,i],np.sum(np.square(X-np.tile(Mu[i:i+1,:],(n,1))),axis=1))*1.0/(d*n_hat[i])
        #compute loglikelihood function
        curCost=0
        for i in xrange(n):
            curCost+=np.log(np.sum(np.multiply(P,normalDensity(X[i:i+1,:],K,Mu,Var))))
        print curCost
    return (Mu,P,Var,post)

# initialization for mixture models
# input: X: n*d data matrix; K: number of mixtures;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
def init(X,K):
    n=len(X)
    d=len(X[0])
    Mu=X[:K,]
    P=1.0/K*np.ones((K,1))
    totalVar=LA.norm(X-np.tile(np.mean(X,axis=0),(n,1)),'fro')**2/(d*n)
    Var=totalVar*np.ones((K,1))
    return (Mu,P,Var)

# initialization for mixture models
# input: X: n*d data matrix; K: number of mixtures;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
def init_student(X,K):
    # WRITE YOUR CODE HERE
    return (Mu,P,Var)

# mixture Guassian
# input: X: n*d data matrix; K: number of mixtures; Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def mixGuass(X,K,Mu,P,Var):
    # WRITE YOUR CODE HERE
    return (Mu,P,Var,post,LL)

#mixture Guassian with missing entries
# input: X: n*d data matrix; K: number of mixtures; Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def mixGaussMiss(X,K,Mu,P,Var):
    # WRITE YOUR CODE HERE
    return (Mu,P,Var,post)

# Bayesian Information Criterion (BIC) for selecting the number of mixture components
# input: X: n*d data matrix; Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: K: number of mixtures;
def BICmix(X,Mu,P,Var):
    # WRITE YOUR CODE HERE
    return K

