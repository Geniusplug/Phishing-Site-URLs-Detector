from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix

def metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else 0,'f1':f1_score(y,pred,zero_division=0),'mcc':matthews_corrcoef(y,pred),'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'fnr':fn/(fn+tp) if fn+tp else 0,'fpr':fp/(fp+tn) if fp+tn else 0,'tp':tp,'tn':tn,'fp':fp,'fn':fn}
