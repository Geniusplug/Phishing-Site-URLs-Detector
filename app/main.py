from __future__ import annotations
from fastapi.templating import Jinja2Templates
import hashlib,json,sqlite3,time,re,ipaddress
from pathlib import Path
from urllib.parse import urlparse
import numpy as np, joblib
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'models'; LOGDIR=ROOT/'logs'; DB=LOGDIR/'prediction_logs.sqlite3'; LOGDIR.mkdir(exist_ok=True)
app=FastAPI(title='RACE-Phish Real-Time URL Analyzer',version='1.1')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static'); templates=Jinja2Templates(directory=ROOT/'templates')
model=joblib.load(MODELS/'deployment_url_model.joblib')
deploy_cols=json.loads((MODELS/'deployment_features.json').read_text())
class URLRequest(BaseModel): url:str=Field(min_length=4,max_length=2048)

def init_db():
    con=sqlite3.connect(DB); con.execute("CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,url_hash TEXT,risk REAL,confidence REAL,decision TEXT,latency_ms REAL,model_version TEXT)"); con.commit(); con.close()
init_db()

def raw_features(url):
    u=url.strip(); parsed=urlparse(u if '://' in u else 'http://'+u); host=parsed.hostname or ''; path=parsed.path or ''; query=parsed.query or ''; host_l=host.lower(); u_l=u.lower()
    try: ip=int(ipaddress.ip_address(host) is not None)
    except Exception: ip=0
    words=[w for w in re.split(r'[^A-Za-z0-9]+',u) if w]
    host_words=[w for w in re.split(r'[^A-Za-z0-9]+',host) if w]
    path_words=[w for w in re.split(r'[^A-Za-z0-9]+',path) if w]
    vals={c:0.0 for c in deploy_cols}
    def put(k,v):
        if k in vals: vals[k]=float(v)
    put('length_url',len(u)); put('length_hostname',len(host)); put('ip',ip); put('nb_dots',u.count('.')); put('nb_hyphens',u.count('-')); put('nb_at',u.count('@')); put('nb_qm',u.count('?')); put('nb_and',u.count('&')); put('nb_or',u.count('|')); put('nb_eq',u.count('=')); put('nb_underscore',u.count('_')); put('nb_tilde',u.count('~')); put('nb_percent',u.count('%')); put('nb_slash',u.count('/')); put('nb_star',u.count('*')); put('nb_colon',u.count(':')); put('nb_comma',u.count(',')); put('nb_semicolumn',u.count(';')); put('nb_dollar',u.count('$')); put('nb_space',u.count(' ')); put('nb_www',int('www' in host_l)); put('nb_com',int('.com' in host_l)); put('nb_dslash',u.count('//')-int('://' in u)); put('http_in_path',int('http' in path.lower())); put('https_token',int('https' in host_l or 'https' in path.lower())); put('ratio_digits_url',sum(ch.isdigit() for ch in u)/max(1,len(u))); put('ratio_digits_host',sum(ch.isdigit() for ch in host)/max(1,len(host))); put('punycode',int('xn--' in host_l)); put('port',int(parsed.port is not None) if parsed.port else 0); put('tld_in_path',int(any(path.lower().endswith(t) or ('/'+t+'/') in path.lower() for t in ['.com','.net','.org','.co','.info']))); put('tld_in_subdomain',int(any(x in host_l.split('.')[:-2] for x in ['com','net','org','co']))); put('abnormal_subdomain',int(host.count('.')>3)); put('nb_subdomains',max(0,host.count('.')-1)); put('prefix_suffix',int('-' in host)); put('random_domain',int(bool(re.search(r'[a-z0-9]{18,}',host_l)))); put('path_extension',int(bool(re.search(r'\.[a-zA-Z0-9]{1,6}$',path)))); put('length_words_raw',len(words)); put('char_repeat',max([u.count(ch) for ch in set(u)] or [0])); put('shortest_words_raw',min([len(w) for w in words] or [0])); put('shortest_word_host',min([len(w) for w in host_words] or [0])); put('shortest_word_path',min([len(w) for w in path_words] or [0])); put('longest_words_raw',max([len(w) for w in words] or [0])); put('longest_word_host',max([len(w) for w in host_words] or [0])); put('longest_word_path',max([len(w) for w in path_words] or [0])); put('avg_words_raw',np.mean([len(w) for w in words]) if words else 0); put('avg_word_host',np.mean([len(w) for w in host_words]) if host_words else 0); put('avg_word_path',np.mean([len(w) for w in path_words]) if path_words else 0)
    return np.array([[vals[c] for c in deploy_cols]],dtype=np.float32)

def recommendation(decision):
    if decision=='PHISHING': return 'Avoid credentials, payment details and downloads. Verify the domain using an independently known official source.'
    if decision=='UNCERTAIN': return 'The model is not sufficiently certain. Treat the URL cautiously and verify the domain independently.'
    return 'No strong phishing signal was detected by this local model. Still verify the domain before entering sensitive information.'
@app.get('/',response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html'
    )
@app.get('/health')
def health(): return {'status':'ok','model':'RACE-Phish URL-only deployment model','features':len(deploy_cols)}
@app.post('/api/analyze')
def analyze(req:URLRequest):
    start=time.perf_counter(); x=raw_features(req.url); risk=float(model.predict_proba(x)[0,1]); conf=max(risk,1-risk)
    decision='PHISHING' if risk>=0.5 else 'LEGITIMATE'; uncertainty=conf<0.72
    if uncertainty: decision='UNCERTAIN'
    ms=(time.perf_counter()-start)*1000; h=hashlib.sha256(req.url.encode()).hexdigest()[:16]
    con=sqlite3.connect(DB); con.execute('INSERT INTO predictions(ts,url_hash,risk,confidence,decision,latency_ms,model_version) VALUES(datetime(\'now\'),?,?,?,?,?,?)',(h,risk,conf,decision,ms,'url-only-v1')); con.commit(); con.close()
    return {'url':req.url,'risk_probability':risk,'confidence':conf,'decision':decision,'latency_ms':ms,'recommendation':recommendation(decision),'evidence':{'raw_url_lexical':'analysed','domain_syntax':'analysed','path/query':'analysed','external_reputation':'not queried in offline mode'},'privacy':'Only a truncated SHA-256 URL hash is stored in the local log; the raw URL is not logged.'}
