# Research protocol

Primary split: 70/15/15 stratified train/validation/test after URL-level duplicate removal.

Baselines: Logistic Regression, Decision Tree, Random Forest, Extra Trees, RBF-SVM.

Proposed model: four semantic MLP evidence experts + entropy reliability + learned fusion + adaptive early-exit inference.

Ablation: view combinations, reliability removal proxy, view dropout.

Robustness: controlled feature-space perturbation proxies. These are not substitutes for a true external live-web adversarial evaluation and should be described accurately in the paper.

The test set is never used for fitting model parameters. All final paper numbers should be generated from a clean final run and archived with the code/configuration.
