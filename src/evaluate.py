from __future__ import annotations
import argparse,joblib,pickle,json,time
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix,roc_auc_score,average_precision_score
from sklearn.calibration import calibration_curve
from .config import *
from .metrics import metrics

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA)); a=p.parse_args(); pred=pd.read_csv(CSV/'test_predictions.csv'); y=pred.y_true.values; pcols=[c for c in pred if c.startswith('p_')]
 rows=[]
 for c in pcols: rows.append({'model':c[2:],'roc_auc':roc_auc_score(y,pred[c]),'pr_auc':average_precision_score(y,pred[c])})
 pd.DataFrame(rows).to_csv(CSV/'auc_summary.csv',index=False)
 # ROC
 plt.figure(figsize=(8,6));
 for c in pcols:
  fpr,tpr,_=roc_curve(y,pred[c]); plt.plot(fpr,tpr,label=f'{c[2:]} (AUC={roc_auc_score(y,pred[c]):.3f})')
 plt.plot([0,1],[0,1],'--',label='Chance'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve Comparison'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG/'fig_roc_comparison.png',dpi=400); plt.savefig(FIG/'fig_roc_comparison.pdf',bbox_inches='tight'); plt.close()
 # PR
 plt.figure(figsize=(8,6));
 for c in pcols:
  pr,re,_=precision_recall_curve(y,pred[c]); plt.plot(re,pr,label=f'{c[2:]} (AP={average_precision_score(y,pred[c]):.3f})')
 plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Comparison'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG/'fig_pr_comparison.png',dpi=400); plt.savefig(FIG/'fig_pr_comparison.pdf',bbox_inches='tight'); plt.close()
 # Confusion matrix proposed
 c='p_race_phish'; cm=confusion_matrix(y,(pred[c]>=.5).astype(int)); plt.figure(figsize=(5.5,4.8)); sns.heatmap(cm,annot=True,fmt='d',cbar=False,xticklabels=['Legitimate','Phishing'],yticklabels=['Legitimate','Phishing']); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('RACE-Phish Confusion Matrix'); plt.tight_layout(); plt.savefig(FIG/'fig_confusion_matrix.png',dpi=400); plt.savefig(FIG/'fig_confusion_matrix.pdf',bbox_inches='tight'); plt.close()
 # Calibration
 plt.figure(figsize=(7,6));
 for c in pcols:
  frac,mean=calibration_curve(y,pred[c],n_bins=10,strategy='quantile'); plt.plot(mean,frac,marker='o',label=c[2:])
 plt.plot([0,1],[0,1],'--',label='Perfect calibration'); plt.xlabel('Mean predicted probability'); plt.ylabel('Observed frequency'); plt.title('Calibration Comparison'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG/'fig_calibration.png',dpi=400); plt.savefig(FIG/'fig_calibration.pdf',bbox_inches='tight'); plt.close()
 # Threshold
 c='p_race_phish'; rows=[]
 for th in np.arange(.05,.96,.05):
  m=metrics(y,pred[c].values,th); rows.append({'threshold':th,**{k:m[k] for k in ['precision','recall','specificity','f1','fpr','fnr']}})
 pd.DataFrame(rows).to_csv(CSV/'threshold_analysis.csv',index=False); tdf=pd.DataFrame(rows); plt.figure(figsize=(8,5));
 for k in ['precision','recall','f1','specificity']: plt.plot(tdf.threshold,tdf[k],marker='o',label=k)
 plt.xlabel('Decision threshold'); plt.ylabel('Metric'); plt.title('Threshold Sensitivity'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_threshold_analysis.png',dpi=400); plt.savefig(FIG/'fig_threshold_analysis.pdf',bbox_inches='tight'); plt.close()
 # model comparison
 mc=pd.read_csv(CSV/'model_comparison.csv'); metrics_cols=['accuracy','f1','mcc','roc_auc','pr_auc','specificity','recall']; long=mc[['model']+metrics_cols].melt('model',var_name='metric',value_name='value'); plt.figure(figsize=(12,6)); sns.barplot(data=long,x='metric',y='value',hue='model'); plt.ylim(0,1.05); plt.xticks(rotation=20); plt.title('Model Performance Comparison'); plt.tight_layout(); plt.savefig(FIG/'fig_model_comparison.png',dpi=400); plt.savefig(FIG/'fig_model_comparison.pdf',bbox_inches='tight'); plt.close()
 # RF feature importance
 rfpath=MODELS/'baselines'/'randomforest.joblib';
 if rfpath.exists():
  rf=joblib.load(rfpath); meta=json.loads((MODELS/'metadata.json').read_text()); fi=pd.DataFrame({'feature':meta['features'],'importance':rf.feature_importances_}).sort_values('importance',ascending=False); fi.to_csv(CSV/'feature_importance.csv',index=False); top=fi.head(20).sort_values('importance'); plt.figure(figsize=(8,7)); plt.barh(top.feature,top.importance); plt.xlabel('Importance'); plt.title('Top 20 Feature Importance (Random Forest)'); plt.tight_layout(); plt.savefig(FIG/'fig_feature_importance.png',dpi=400); plt.savefig(FIG/'fig_feature_importance.pdf',bbox_inches='tight'); plt.close()

 # Training curves
 hist=CSV/'training_history.csv'
 if hist.exists():
  h=pd.read_csv(hist); plt.figure(figsize=(9,5));
  for view,g in h.groupby('view'): plt.plot(g.epoch,g.val_loss,label=view)
  plt.xlabel('Epoch'); plt.ylabel('Validation loss'); plt.title('RACE-Phish Training Curves'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'fig_training_curves.png',dpi=400); plt.savefig(FIG/'fig_training_curves.pdf',bbox_inches='tight'); plt.close()
 # Ablation visual
 ab=CSV/'ablation_study.csv'
 if ab.exists():
  ad=pd.read_csv(ab); plt.figure(figsize=(10,5)); plt.bar(ad.experiment,ad.f1); plt.xticks(rotation=35,ha='right'); plt.ylabel('F1'); plt.ylim(0,1.05); plt.title('Component-wise Ablation Study'); plt.tight_layout(); plt.savefig(FIG/'fig_ablation_study.png',dpi=400); plt.savefig(FIG/'fig_ablation_study.pdf',bbox_inches='tight'); plt.close()
 # Robustness visual
 rb=CSV/'robustness_results.csv'
 if rb.exists():
  rd=pd.read_csv(rb); plt.figure(figsize=(9,5)); plt.bar(rd.perturbation,rd.mean_abs_probability_shift); plt.xticks(rotation=25,ha='right'); plt.ylabel('Mean absolute probability shift'); plt.title('Robustness to Controlled Perturbations'); plt.tight_layout(); plt.savefig(FIG/'fig_robustness.png',dpi=400); plt.savefig(FIG/'fig_robustness.pdf',bbox_inches='tight'); plt.close()
 # All-model confusion matrices as a compact CSV table
 cms=[]
 for c in pcols:
  tn,fp,fn,tp=confusion_matrix(y,(pred[c]>=.5).astype(int),labels=[0,1]).ravel(); cms.append({'model':c[2:],'tn':tn,'fp':fp,'fn':fn,'tp':tp})
 pd.DataFrame(cms).to_csv(CSV/'confusion_matrices.csv',index=False)
 print('Evaluation complete. Figures:',FIG); print('CSV:',CSV)
if __name__=='__main__': main()
