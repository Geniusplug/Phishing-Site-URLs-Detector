"""One-command research pipeline."""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args):
    print("\n>>> " + " ".join(args), flush=True)
    subprocess.run(args,cwd=ROOT,check=True)
if __name__=='__main__':
    py=sys.executable; data=str(ROOT/'data'/'dataset_phishing.csv'); epochs=sys.argv[1] if len(sys.argv)>1 else '100'
    run([py,'-m','src.train','--data',data,'--epochs',epochs])
    run([py,'-m','src.ablation','--data',data,'--epochs','40'])
    run([py,'-m','src.robustness','--data',data,'--n','300'])
    run([py,'-m','src.evaluate','--data',data])
    run([py,'-m','src.statistics'])
    print("\nALL EXPERIMENTS COMPLETE. See results/csv and results/figures.")
