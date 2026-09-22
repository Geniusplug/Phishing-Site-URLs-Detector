from __future__ import annotations
import argparse,time,pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score
from .config import *
from .features import infer_feature_views,load_numeric_matrix
from .race_model import RACEPhish
from .seed import seed_everything

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); p.add_argument('--epochs',type=int,default=40); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); seed_everything(a.seed)
    df=pd.read_csv(a.data).drop_duplicates('url').reset_index(drop=True); y=(df.status.str.lower()=='phishing').astype(int).values; cols,views=infer_feature_views(df); X=load_numeric_matrix(df,cols)
    idx=np.arange(len(df)); tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=a.seed); va,te=train_test_split(tmp,test_size=.50,stratify=y[tmp],random_state=a.seed); sc=StandardScaler().fit(X[tr]); X=sc.transform(X).astype('float32')
    def inds(keys): return [[cols.index(c) for c in views[k] if c in cols] for k in keys]
    configs=[('Full-RACE',['lexical','domain','path','semantic'],False,True,True),('No-Reliability',['lexical','domain','path','semantic'],False,False,True),('No-Consistency',['lexical','domain','path','semantic'],False,True,False),('Lexical-only',['lexical'],False,True,True),('Lexical+Domain',['lexical','domain'],False,True,True),('Lexical+Domain+Semantic',['lexical','domain','semantic'],False,True,True),('View-Dropout',['lexical','domain','path','semantic'],True,True,True)]
    rows=[]
    for name,keys,drop,use_rel,use_cons in configs:
        print(f"\n[ABLATION] {name}",flush=True)
        model=RACEPhish(inds(keys),seed=a.seed,epochs=a.epochs,hidden=20,lr=.002,dropout_views=drop,use_reliability=use_rel,use_consistency=use_cons)
        t=time.perf_counter(); model.fit(X[tr],y[tr],X[va],y[va],verbose=False); sec=time.perf_counter()-t
        p=model.predict_proba(X[te])[:,1]
        rows.append({'experiment':name,'views':'+'.join(keys),'view_dropout':drop,'use_reliability':use_rel,'use_consistency':use_cons,'f1':f1_score(y[te],p>=.5),'roc_auc':roc_auc_score(y[te],p),'train_seconds':sec,'epochs':a.epochs})
    out=pd.DataFrame(rows); out.to_csv(CSV/'ablation_study.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
