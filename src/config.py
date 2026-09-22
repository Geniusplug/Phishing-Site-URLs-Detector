from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'dataset_phishing.csv'
CHECK=RESULTS/'checkpoints'
MODELS=ROOT/'models'
BASELINES=MODELS/'baselines'
for p in [RESULTS,CSV,FIG,CHECK,MODELS,BASELINES]: p.mkdir(parents=True,exist_ok=True)
