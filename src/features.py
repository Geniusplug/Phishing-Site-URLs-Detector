from __future__ import annotations
import numpy as np, pandas as pd
FEATURE_COLUMNS=[]
VIEW_COLUMNS={"lexical":[],"domain":[],"path":[],"semantic":[]}

def infer_feature_views(df):
    cols=[c for c in df.columns if c not in ('url','status')]
    # Stable semantic groups based on feature names; remaining features go to semantic.
    lexical_keys=('length_url','nb_','ratio_','shortest_words','longest_words','avg_words','char_repeat','prefix_suffix','num_')
    domain_keys=('hostname','subdomain','domain','tld','whois','ip','punycode','registration','dns','brand')
    path_keys=('path','directory','file','query','url_of_anchor','link','slash','percent','at','qm','and','eq')
    semantic_keys=('google','page_rank','web_traffic','statistical','safe','phish','login','favicon','iframe','popup','right_clic','onmouseover','hyperlink','external','submit_email','sfh','redirect','empty_title')
    groups={k:[] for k in VIEW_COLUMNS}
    used=set()
    for c in cols:
        lc=c.lower()
        if any(k in lc for k in domain_keys) and not any(k in lc for k in ('url_of_anchor','ratio_ext','ratio_int')):
            groups['domain'].append(c); used.add(c)
        elif any(k in lc for k in semantic_keys):
            groups['semantic'].append(c); used.add(c)
        elif any(k in lc for k in path_keys) and not any(k in lc for k in ('domain','hostname')):
            groups['path'].append(c); used.add(c)
        elif any(k in lc for k in lexical_keys):
            groups['lexical'].append(c); used.add(c)
    for c in cols:
        if c not in used: groups['semantic'].append(c)
    # guarantee non-empty groups by distributing remaining columns if necessary
    for k in groups:
        if not groups[k]: groups[k]=cols[:max(1,len(cols)//4)]
    return cols,groups

def load_numeric_matrix(df, feature_cols):
    X=df[feature_cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.astype(np.float32).values
