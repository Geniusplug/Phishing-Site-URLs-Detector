import os,random,numpy as np
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed)
