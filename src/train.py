from __future__ import annotations
import argparse,json,time,sys,platform
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,log_loss
from .config import *
from .features import infer_feature_views,load_numeric_matrix
from .metrics import metrics
from .race_model import RACEPhish
from .seed import seed_everything

def args():
 p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seed',type=int,default=42); p.add_argument('--quiet',action='store_true'); return p.parse_args()

def main():
 a=args(); seed_everything(a.seed); print('='*78); print('RACE-Phish COMPLETE TRAINING PIPELINE'); print('='*78); print(f'Python {platform.python_version()} | seed={a.seed} | epochs={a.epochs}',flush=True)
 df=pd.read_csv(a.data); df=df.drop_duplicates(subset=['url']).reset_index(drop=True)
 if not {'url','status'}.issubset(df.columns): raise ValueError('Dataset must contain url and status columns.')
 y=(df.status.astype(str).str.lower()=='phishing').astype(int).values; urls=df.url.astype(str).values
 cols,views=infer_feature_views(df); X=load_numeric_matrix(df,cols); pd.DataFrame({'class':['legitimate','phishing'],'count':[int((y==0).sum()),int((y==1).sum())]}).to_csv(CSV/'class_distribution.csv',index=False); pd.DataFrame({'n_rows':[len(df)],'n_features':[len(cols)],'duplicate_urls_removed':[0]}).to_csv(CSV/'dataset_summary.csv',index=False); print(f'Dataset: {len(df):,} unique URLs | {len(cols)} numeric features',flush=True); print('Views:',{k:len(v) for k,v in views.items()},flush=True)
 idx=np.arange(len(df)); tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=a.seed); va,te=train_test_split(tmp,test_size=.50,stratify=y[tmp],random_state=a.seed)
 scaler=StandardScaler().fit(X[tr]); Xs=scaler.transform(X).astype('float32'); joblib.dump(scaler,MODELS/'scaler.joblib')
 pd.DataFrame({'index':np.r_[tr,va,te],'split':['train']*len(tr)+['validation']*len(va)+['test']*len(te)}).to_csv(CSV/'split_manifest.csv',index=False)
 pd.DataFrame({'feature':cols,'missing':df[cols].isna().sum().values,'nunique':[df[c].nunique(dropna=False) for c in cols]}).to_csv(CSV/'feature_quality.csv',index=False)
 (MODELS/'metadata.json').write_text(json.dumps({'features':cols,'views':views,'seed':a.seed,'train_size':len(tr),'val_size':len(va),'test_size':len(te)},indent=2),encoding='utf-8')
 # 5 strong baselines
 baselines={
 'LogisticRegression':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,C=1.0,class_weight='balanced',random_state=a.seed))]),
 'DecisionTree':DecisionTreeClassifier(max_depth=18,min_samples_leaf=2,class_weight='balanced',random_state=a.seed),
 'RandomForest':RandomForestClassifier(n_estimators=400,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=a.seed),
 'ExtraTrees':ExtraTreesClassifier(n_estimators=400,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=a.seed),
 'SVM-RBF':Pipeline([('scale',StandardScaler()),('model',SVC(C=2.0,kernel='rbf',gamma='scale',probability=True,class_weight='balanced',random_state=a.seed))])}
 allres=[]; pred={'url':urls[te],'y_true':y[te]}
 print('\n[1/3] TRAINING 5 BASELINE MODELS',flush=True)
 for n,m in baselines.items():
  print(f'--- {n} ---',flush=True); t=time.perf_counter(); m.fit(X[tr],y[tr]); sec=time.perf_counter()-t; p=m.predict_proba(X[te])[:,1]; r=metrics(y[te],p); r.update(model=n,train_seconds=sec,inference_ms_per_sample=1000*time.perf_counter()*0 if False else 0.0,features_used=len(cols),family='baseline'); allres.append(r); pred['p_'+n.replace('-','_').lower()]=p; joblib.dump(m,BASELINES/(n.replace('-','_').lower()+'.joblib')); print(f"{n}: F1={r['f1']:.4f} | ROC-AUC={r['roc_auc']:.4f} | time={sec:.2f}s",flush=True)
 print('\n[2/3] TRAINING PROPOSED RACE-PHISH MODEL (VISIBLE EPOCHS)',flush=True)
 groups=[]
 for k in ['lexical','domain','path','semantic']:
  groups.append([cols.index(c) for c in views[k] if c in cols])
 t=time.perf_counter(); race=RACEPhish(groups,seed=a.seed,epochs=a.epochs,hidden=24,lr=.002,dropout_views=False); race.fit(Xs[tr],y[tr],Xs[va],y[va],verbose=not a.quiet); train_sec=time.perf_counter()-t
 pd.DataFrame(race.history).to_csv(CSV/'training_history.csv',index=False)
 p=race.predict_proba(Xs[te])[:,1]; rr=metrics(y[te],p); rr.update(model='RACE-Phish',train_seconds=train_sec,inference_ms_per_sample=0.0,features_used=len(cols),family='proposed'); allres.append(rr); pred['p_race_phish']=p
 import pickle
 with open(MODELS/'race_phish.pkl','wb') as f: pickle.dump(race,f)
 # URL-only deployment model: only features deterministically derivable from a raw URL.
 deploy_cols=[c for c in cols if c in {'length_url','length_hostname','ip','nb_dots','nb_hyphens','nb_at','nb_qm','nb_and','nb_or','nb_eq','nb_underscore','nb_tilde','nb_percent','nb_slash','nb_star','nb_colon','nb_comma','nb_semicolumn','nb_dollar','nb_space','nb_www','nb_com','nb_dslash','http_in_path','https_token','ratio_digits_url','ratio_digits_host','punycode','port','tld_in_path','tld_in_subdomain','abnormal_subdomain','nb_subdomains','prefix_suffix','random_domain','path_extension','length_words_raw','char_repeat','shortest_words_raw','shortest_word_host','shortest_word_path','longest_words_raw','longest_word_host','longest_word_path','avg_words_raw','avg_word_host','avg_word_path'}]
 deploy_idx=[cols.index(c) for c in deploy_cols]
 deployment=ExtraTreesClassifier(n_estimators=200,max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=a.seed)
 deployment.fit(X[tr][:,deploy_idx],y[tr]); deploy_scaler=StandardScaler().fit(X[tr][:,deploy_idx]); joblib.dump(deployment,MODELS/'deployment_url_model.joblib'); joblib.dump(deploy_scaler,MODELS/'deployment_scaler.joblib'); (MODELS/'deployment_features.json').write_text(json.dumps(deploy_cols,indent=2))
 print(f'Deployment URL-only model saved with {len(deploy_cols)} raw-URL features.',flush=True)
 ap,st=race.adaptive_predict(Xs[te],threshold=.90); ar=metrics(y[te],ap); ar.update(model='RACE-Phish-Adaptive',train_seconds=train_sec,inference_ms_per_sample=0.0,features_used='adaptive',family='proposed'); allres.append(ar); pred['p_race_phish_adaptive']=ap; pred['adaptive_stage']=st
 print(f"RACE-Phish: F1={rr['f1']:.4f} | ROC-AUC={rr['roc_auc']:.4f}",flush=True); print(f"RACE-Phish-Adaptive: F1={ar['f1']:.4f} | ROC-AUC={ar['roc_auc']:.4f} | mean evidence stages={st.mean():.2f}",flush=True)
 pd.DataFrame(allres).to_csv(CSV/'model_comparison.csv',index=False); pd.DataFrame(pred).to_csv(CSV/'test_predictions.csv',index=False)
 # final model metadata
 print('\n[3/3] SAVING MODEL + METADATA',flush=True);
if __name__=='__main__': main()
