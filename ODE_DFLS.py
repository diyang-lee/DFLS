import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import block_diag

from sklearn.metrics import mean_squared_error
from group_lasso import GroupLasso
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import time



def hf(X,w): #Hypothesis Function
    return 1./(1+np.exp(-X@w))

def sig(x):
    return 1./(1+np.exp(-x))

def S(x):
    return sig(x)*(1-sig(x))
    

eps=1e-8 
def increFeature_GL(X,y,x_new,w0,group1,group2,lam,nn):
    wt=w0.copy()
    wt_p=wt # previous w
    active1=[]
    active2=[]
    active_g1=[]
    active_g2=[]
    tmp=0
    n=X.shape[0]
    for ele in group1:
        if ele[0]==1:
            active_g1.append(np.arange(tmp,tmp+len(ele[1:])))
            tmp=tmp+len(ele[1:])
            active1+=ele[1:]
    tp_0=[]
    X_=np.concatenate((X,x_new),axis=1) # whole matrix
    for ele in group2:
        tp_0.append(np.sqrt(len(ele[1:]))/(np.linalg.norm(X_[:,ele[1:]].T@(y-X@w0[:group2[0][1]]))))
    tp=np.min(tp_0)*2.*n*lam/nn
    #print("first seg=",tp)
    if tp>(1./nn):
        return w0
    idx=np.argmin(tp_0)
    group2[idx][0]=1
    for ele in group2:
        if ele[0]==1:
            active_g2.append(np.arange(tmp,tmp+len(ele[1:])))
            tmp=tmp+len(ele[1:])
            active2+=ele[1:]
    wt_=tp*X_[:,group2[idx][1:]].T@(y-X@w0[:group2[0][1]])
    wt[group2[idx][1:]]=(wt_.flatten())*eps
    def func(theta,w_A):
        w_A=w_A.reshape(-1,1)
        diag1,diag2=[],[]
        for ele in active_g1:
            wk=np.sqrt(len(ele))
            t1=w_A[ele].reshape(-1,1)
            norms=np.linalg.norm(t1)
            a1=wk*(norms**2*np.eye(len(ele))-t1@(t1.T))/(norms**3)
            diag1.append(a1)
        for ele in active_g2:
            wk=np.sqrt(len(ele))
            t1=w_A[ele].reshape(-1,1)
            norms=np.linalg.norm(t1)
            a1=wk*(norms**2*np.eye(len(ele))-t1@(t1.T))/(norms**3)
            diag2.append(a1)
            
        d2R_1,d2R_2=block_diag(*diag1),block_diag(*diag2)
        A=X_A.T@X_A/n/2+lam*d2R_1
        B=nn*theta*x_new_A.T@X_A/n/2
        C=((nn*theta)**2)*x_new_A.T@x_new_A/n/2+lam*d2R_2
        AA=np.block([[A,B.T],[B,C]])
        b1=X_A.T@x_new_A@w_A[active_g2[0][0]:]/n/2
        b2=x_new_A.T@(X_A@w_A[:active_g2[0][0]]+2*nn*theta*x_new_A@w_A[active_g2[0][0]:]-y.reshape(-1,1))/n/2
        beta=nn*np.concatenate((b1,b2),axis=0)
        ans=-np.linalg.inv(AA)@beta
        return ans.flatten()
    Ns=175
    # tuning parameter control accuracy of segmentations
    ds=((1./nn)-tp)/Ns
    for s in np.linspace(tp,(1./nn),Ns+1)[:-1]:
        X_A=X[:,active1]
        x_new_A=X_[:,active2]
        w_A=np.concatenate((wt[active1],wt[active2]),axis=0)
        wt_A=odeint(func,w_A,[s,s+ds],tfirst=True)[-1]
        wt=np.zeros(len(w0))
        wt[active1+active2]=wt_A
        temp=0
        active1=[]
        active2=[]
        active_g1=[]
        active_g2=[]
        delta1=(y-X@wt_p[:group2[0][1]]-nn*s*x_new@wt_p[group2[0][1]:]).reshape(-1,1)
        delta2=(y-X@wt[:group2[0][1]]-nn*(s+ds)*x_new@wt[group2[0][1]:]).reshape(-1,1)
        for ele in group1:
            if ele[0]==1:
                to_inactive=True
                for j in ele[1:]:
                    if wt_p[j]*wt[j]>0:
                        to_inactive=False
                if to_inactive :
                    #print("theta=",s,',turn to inactive',ele)  
                    ele[0]=0
                else:
                    active_g1.append(np.arange(temp,temp+len(ele[1:])))
                    temp=temp+len(ele[1:])
                    active1+=ele[1:]
            elif ele[0]==0:
                wk=np.sqrt(len(ele[1:]))
                tmp=np.linalg.norm(X_[:,ele[1:]].T@delta1) /2/n-lam*wk
                tmp2=np.linalg.norm(X_[:,ele[1:]].T@delta2) /2/n-lam*wk
                if tmp*tmp2<0:
                    #print("theta=",s,',turn to active',ele)                        
                    ele[0]=1
                    wt_=X_[:,ele[1:]].T@delta1
                    wt[ele[1:]]=(wt_.flatten())*eps
                    active_g1.append(np.arange(temp,temp+len(ele[1:])))
                    temp=temp+len(ele[1:])
                    active1+=ele[1:]
        for ele in group2:
            if ele[0]==1:
                to_inactive=True
                for j in ele[1:]:
                    if wt_p[j]*wt[j]>0:
                        to_inactive=False
                if to_inactive :
                #if wt_p[ele[1]]*wt[ele[1]]<0:
                    #print("theta=",s,',turn to inactive',ele) 
                    ele[0]=0
                else:
                    active_g2.append(np.arange(temp,temp+len(ele[1:])))
                    temp=temp+len(ele[1:])
                    active2+=ele[1:]
            elif ele[0]==0:
                wk=np.sqrt(len(ele[1:]))
                tmp=s*np.linalg.norm(X_[:,ele[1:]].T@delta1) /2/n-lam*wk
                tmp2=(s+ds)*np.linalg.norm(X_[:,ele[1:]].T@delta2) /2/n-lam*wk
                if tmp*tmp2<0:
                    #print("theta=",s,',turn to active',ele)                        
                    ele[0]=1
                    wt_=X_[:,ele[1:]].T@delta1
                    wt[ele[1:]]=(wt_.flatten())*eps
                    active_g2.append(np.arange(temp,temp+len(ele[1:])))
                    temp=temp+len(ele[1:])
                    active2+=ele[1:]
        wt_p=wt.copy()
    return wt,tp
    


def increFeature_logit(X,y,x_new,w0,group1,group2,C,nn):
    wt=w0.copy()
    def func(theta,w):
        w=w.reshape(-1,1)
        u=X@w[:group2[0]]+nn*theta*x_new@w[group2[0]:]
        u=u.reshape(-1,1)
        A=C*X.T@(S(u)*X)+np.eye(len(group1))
        B=C*nn*theta*x_new.T@(S(u)*X)
        D=C*(nn*theta)**2*x_new.T@(S(u)*x_new)+np.eye(len(group2))
        AA=np.block([[A,B.T],[B,D]])
        u_=(x_new@w[group2[0]:]).reshape(-1,1)
        b1=C*X.T@(S(u)*u_)
        b2=C*x_new.T@( (sig(u)-y.reshape(-1,1))+nn*theta*(S(u)*u_) )
        beta=nn*np.concatenate((b1,b2),axis=0)
        ans=-np.linalg.inv(AA)@beta
        return ans.flatten()
    

    wt=odeint(func,wt,[0,(1./nn)],tfirst=True)[-1]


    return wt
 

def increFeature_poi(X,y,x_new,w0,group1,group2,lam,nn):
    wt=w0.copy()
    n=X.shape[0] 
    def func(theta,w):
        w=w.reshape(-1,1)
        u=np.exp(X@w[:group2[0]]+nn*theta*x_new@w[group2[0]:])
        u=u.reshape(-1,1)
        A=1./n*X.T@(u*X)+2*lam*np.eye(len(group1))
        B=nn*theta/n*x_new.T@(u*X)
        D=(nn*theta)**2/n*x_new.T@(u*x_new)+2*lam*np.eye(len(group2))
        AA=np.block([[A,B.T],[B,D]])
        u_=(x_new@w[group2[0]:]).reshape(-1,1)
        b1=X.T@(u*u_)
        b2=x_new.T@( (u-y.reshape(-1,1))+nn*theta*(u*u_) )
        beta=nn/n*np.concatenate((b1,b2),axis=0)
        ans=-np.linalg.inv(AA)@beta
        return ans.flatten()
    

    wt=odeint(func,wt,[0,(1./nn)],tfirst=True,h0=0.1/nn,rtol=1e-7, atol=1e-6)[-1]

    return wt
    
def increFeature_gamma(X,y,x_new,w0,group1,group2,lam,nn):
    wt=w0.copy()
    def func(theta,w):
        w=w.reshape(-1,1)
        n=X.shape[0]

        u=np.exp(X@w[:group2[0]]+nn*theta*x_new@w[group2[0]:])
        u=y.reshape(-1,1)/u.reshape(-1,1)
        A=1./n*X.T@(u*X)+2*lam*np.eye(len(group1))
        B=nn*theta/n*x_new.T@(u*X)
        D=(nn*theta)**2/n*x_new.T@(u*x_new)+2*lam*np.eye(len(group2))
        AA=np.block([[A,B.T],[B,D]])
        u_=(x_new@w[group2[0]:]).reshape(-1,1)
        b1=X.T@(u*u_)
        b2=x_new.T@( (1-u)+nn*theta*(u*u_) )
        beta=nn/n*np.concatenate((b1,b2),axis=0)
        ans=-np.linalg.inv(AA)@beta
        return ans.flatten()
    
    wt=odeint(func,wt,[0,(1./nn)],tfirst=True)[-1]

    return wt
