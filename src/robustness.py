from __future__ import annotations
import argparse,pandas as pd,numpy as np,pickle
from .config import *
from .features import infer_feature_views,load_numeric_matrix
import joblib

def perturb(u,kind):
 if kind=='add_query': return u + ('&' if '?' in u else '?') + 'utm_source=research'
 if kind=='hyphen': return u.replace('.','-.' ,1) if '.' in u else u+'-secure'
 if kind=='path_padding': return u.rstrip('/')+'/account/login/verify/session'
 if kind=='case': return u.swapcase()
 return u

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); p.add_argument('--n',type=int,default=300); a=p.parse_args(); df=pd.read_csv(a.data).drop_duplicates('url').reset_index(drop=True).head(a.n); cols,_=infer_feature_views(df); X=load_numeric_matrix(df,cols); model=pickle.load(open(MODELS/'race_phish.pkl','rb')); scaler=joblib.load(MODELS/'scaler.joblib'); X=scaler.transform(X).astype('float32'); base=model.predict_proba(X)[:,1]; rows=[]
 # This is a feature-space perturbation proxy; raw URL parser changes are reported as a future deployment experiment.
 for kind in ['add_query','hyphen','path_padding','case']:
  # deterministic feature perturbation proxy: perturb a few lexical/path dimensions where available
  Xp=X.copy();
  for j,c in enumerate(cols):
   if kind=='add_query' and any(s in c.lower() for s in ['qm','and','eq','query']): Xp[:,j]+=1
   elif kind=='hyphen' and 'hyphen' in c.lower(): Xp[:,j]+=1
   elif kind=='path_padding' and 'path' in c.lower(): Xp[:,j]+=1
   elif kind=='case':
    if 'ratio' in c.lower(): Xp[:,j]=np.clip(Xp[:,j]*1.001,0,None)
  pp=model.predict_proba(Xp)[:,1]; rows.append({'perturbation':kind,'n':len(X),'mean_abs_probability_shift':float(np.mean(np.abs(pp-base))),'prediction_flip_rate':float(np.mean((pp>=.5)!=(base>=.5)))})
 pd.DataFrame(rows).to_csv(CSV/'robustness_results.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
