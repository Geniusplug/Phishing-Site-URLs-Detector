from __future__ import annotations
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class RACEPhish:
    """Lightweight reliability-adaptive evidence model.
    Four MLP experts are trained on semantic views; a learned fusion head combines
    expert probabilities and entropy-derived reliability. The training loop uses
    one epoch at a time so VS Code/terminal visibly reports progress.
    """
    names=['lexical','domain','path','semantic']
    def __init__(self,groups,seed=42,epochs=100,hidden=24,lr=0.002,dropout_views=False,use_reliability=True,use_consistency=True):
        self.groups=groups; self.seed=seed; self.epochs=epochs; self.hidden=hidden; self.lr=lr; self.dropout_views=dropout_views; self.use_reliability=use_reliability; self.use_consistency=use_consistency
    @staticmethod
    def reliability(p):
        p=np.clip(p,1e-7,1-1e-7); h=-(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2); return 1-h
    def fit(self,X,y,Xv=None,yv=None,verbose=True):
        self.experts=[]
        self.history=[]
        for i,idx in enumerate(self.groups):
            model=MLPClassifier(hidden_layer_sizes=(self.hidden,),activation='relu',solver='adam',learning_rate_init=self.lr,max_iter=1,warm_start=True,random_state=self.seed+i,batch_size=128,alpha=1e-4)
            for ep in range(1,self.epochs+1):
                rng=np.random.RandomState(self.seed+i*1000+ep)
                Xi=X[:,idx]
                yi=y
                if self.dropout_views and i>0:
                    keep=rng.rand(len(Xi))>0.12
                    if keep.sum()>10: Xi=Xi[keep]; yi=y[keep]
                model.fit(Xi,yi)
                if verbose:
                    p=model.predict_proba(X[:,idx])[:,1]; tr_loss=log_loss(y,p,labels=[0,1])
                    if Xv is not None:
                        pv=model.predict_proba(Xv[:,idx])[:,1]; vl=log_loss(yv,pv,labels=[0,1])
                        print(f"[RACE][{self.names[i]:8s}] Epoch {ep:03d}/{self.epochs} | train_loss={tr_loss:.5f} | val_loss={vl:.5f}",flush=True); self.history.append({'view':self.names[i],'epoch':ep,'train_loss':tr_loss,'val_loss':vl})
                    else: print(f"[RACE][{self.names[i]:8s}] Epoch {ep:03d}/{self.epochs} | train_loss={tr_loss:.5f}",flush=True); self.history.append({'view':self.names[i],'epoch':ep,'train_loss':tr_loss,'val_loss':np.nan})
            self.experts.append(model)
        P=self.expert_probabilities(X)
        if Xv is not None:
            fusion_X=self._features(self.expert_probabilities(Xv)); fusion_y=yv
        else:
            fusion_X=self._features(P); fusion_y=y
        self.fusion=LogisticRegression(max_iter=2000,C=0.5,random_state=self.seed).fit(fusion_X,fusion_y)
        if Xv is not None:
            pv=self.predict_proba(Xv)[:,1]; print(f"[RACE][fusion   ] validation ROC-AUC={__import__('sklearn').metrics.roc_auc_score(yv,pv):.5f}",flush=True)
        return self
    def _features(self,P):
        parts=[P]
        if self.use_reliability:
            R=np.column_stack([self.reliability(P[:,i]) for i in range(P.shape[1])]); parts.append(R)
        if self.use_consistency:
            parts.append(np.std(P,axis=1,keepdims=True))
        return np.column_stack(parts)
    def expert_probabilities(self,X):
        return np.column_stack([m.predict_proba(X[:,idx])[:,1] for m,idx in zip(self.experts,self.groups)])
    def predict_proba(self,X): return self.fusion.predict_proba(self._features(self.expert_probabilities(X)))
    def predict(self,X,threshold=.5): return (self.predict_proba(X)[:,1]>=threshold).astype(int)
    def adaptive_predict(self,X,threshold=.90):
        P=self.expert_probabilities(X); n=len(X); final=np.zeros(n); stages=np.zeros(n,dtype=int); active=np.ones(n,dtype=bool)
        for i in range(P.shape[1]):
            sub=P[:,:i+1]; R=np.column_stack([self.reliability(sub[:,j]) for j in range(i+1)]); W=R/(R.sum(1,keepdims=True)+1e-8) if self.use_reliability else np.ones_like(R)/(i+1); risk=(sub*W).sum(1); disagreement=np.std(sub,axis=1); conf=np.maximum(risk,1-risk)*(1-np.clip(disagreement,0,0.5))
            stop=(conf>=threshold)&active; final[stop]=risk[stop]; stages[stop]=i+1; active[stop]=False
        if active.any(): final[active]=self.predict_proba(X[active])[:,1]; stages[active]=P.shape[1]
        return np.clip(final,0.0,1.0),stages
