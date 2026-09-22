# RACE-Phish model card

## Research model
`models/race_phish.pkl` is the proposed four-view RACE-Phish model trained on the engineered dataset features. It is intended for controlled offline research evaluation.

## Five baseline models
`models/baselines/` contains Logistic Regression, Decision Tree, Random Forest, Extra Trees and RBF-SVM trained on the same locked split.

## Production/offline website model
The website uses `models/deployment_url_model.joblib`. This model is deliberately trained only on 47 features that can be derived deterministically from a raw URL without fetching a page, DNS, WHOIS or external reputation service. This is necessary for a safe offline product. It should not be confused with the full-feature research model.

## Privacy
The API stores a truncated SHA-256 URL hash, risk, confidence, decision and latency in SQLite by default. Raw URLs are not stored.

## Limitations
A raw URL cannot reproduce external page-content, WHOIS, DNS, traffic or reputation features contained in the research dataset. Therefore the offline website is a deployment demonstration, not evidence of full external-world detection coverage.
