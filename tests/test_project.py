from pathlib import Path

def test_structure():
 root=Path(__file__).resolve().parents[1]
 for p in ['data/dataset_phishing.csv','src/train.py','src/race_model.py','app/main.py','templates/index.html']:
  assert (root/p).exists(), p
