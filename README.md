# KNN — Curse of Dimensionality on Sonar Dataset

K-Nearest Neighbors applied to the UCI Sonar dataset (208 samples, 60 features, 2 classes: mine vs rock) to study distance metrics, hyperparameter selection, and the effect of dimensionality reduction.

## Approach
1. Stratified train/test split
2. Baseline KNN (k=5) on raw features, with and without standardization
3. k-sweep (1–30) to observe bias-variance behavior
4. Distance metric comparison: Euclidean / Manhattan / Cosine
5. PCA sweep (2–50 components) → KNN accuracy at each dimensionality
6. Metric comparison again after PCA to 15D

## Results
- Raw 60D accuracy: **0.7381** (identical with/without scaling — Sonar is pre-normalized)
- Peak accuracy: **0.8571 at 15 PCA components** (83.3% variance retained)
- Beyond 20 components, accuracy **degrades below the raw baseline** — the trailing variance directions are noise
- Metric behavior on raw 60D: Manhattan and Cosine beat Euclidean (0.7619 vs 0.7381)
- After PCA to 15D: Euclidean and Cosine converge (both 0.8571); Manhattan trails (0.8095)

## Key Observations
- **Curse of dimensionality is directly visible**: accuracy peaks at intermediate dimensionality, then falls as noisy directions are reintroduced.
- **PCA is Euclidean by construction**: it helps L2-based metrics most. Manhattan benefits less because PCA's objective isn't aligned with its geometry.
- **Standardization is diagnostic, not automatic**: pre-normalized data gains nothing from it.
- **Variance ≠ usefulness** (same lesson as the PCA project): the last 15% of variance actively hurts KNN.

## Stack
`Python` · `scikit-learn` · `numpy` · `pandas` · `matplotlib`

## Run
```bash
pip install numpy pandas matplotlib scikit-learn
python Sonar.py