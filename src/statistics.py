from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from .config import CSV

def bootstrap_auc(y,p,n=2000,seed=42):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); p=np.asarray(p)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y));
        if len(np.unique(y[idx]))<2: continue
        vals.append(roc_auc_score(y[idx],p[idx]))
    return np.percentile(vals,[2.5,50,97.5])

def main():
    df=pd.read_csv(CSV/'test_predictions.csv'); y=df.y_true.values; rows=[]
    for c in [c for c in df if c.startswith('p_')]:
        lo,mid,hi=bootstrap_auc(y,df[c].values); rows.append({'model':c[2:],'roc_auc_bootstrap_median':mid,'ci95_low':lo,'ci95_high':hi,'n_bootstrap':2000})
    pd.DataFrame(rows).to_csv(CSV/'bootstrap_auc_ci.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
