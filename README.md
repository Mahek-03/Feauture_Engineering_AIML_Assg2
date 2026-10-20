# Housing Price Prediction — Feature Engineering

Feature engineering pipeline for the Housing dataset (545 rows, 13 columns), built as part of an AI/ML assignment. Covers cleaning, EDA, encoding, feature creation, train/test splitting without leakage, outlier/skew handling, feature selection, and validation of whether the engineered features actually improve model performance.

## Dataset

`Housing.csv` — 545 rows, 13 columns. Target: `price`.

| Column | Description |
|---|---|
| `price` | Sale price (target) |
| `area` | Plot area (sq. ft.) |
| `bedrooms`, `bathrooms`, `stories` | Counts |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | yes/no amenity flags |
| `parking` | Number of parking spots |
| `furnishingstatus` | furnished / semi-furnished / unfurnished |

No missing values or duplicates in the raw file.

## Project structure

```
.
├── Housing.csv                          # raw dataset
├── Housing_Feature_Engineering.ipynb    # full pipeline (run in Google Colab)
├── housing_engineered_full.csv          # cleaned + encoded + engineered features (generated)
├── housing_train_processed.csv          # processed train split (generated)
├── housing_test_processed.csv           # processed test split (generated)
├── housing_final_model.joblib           # fitted model (generated)
└── README.md
```

## How to run

1. Open `Housing_Feature_Engineering.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run all cells top to bottom.
3. When prompted in the first cell, upload `Housing.csv`.
4. Processed CSVs and the saved model are generated at the end and offered as downloads.

## Pipeline

1. **Understand the data** — shape, dtypes, column groups.
2. **Clean** — missing values, duplicates, inconsistent text, invalid values, IQR outlier detection.
3. **EDA** — distributions, price vs. feature plots, correlation heatmap.
4. **Encode** — yes/no → 1/0; `furnishingstatus` as ordinal (unfurnished < semi < furnished).
5. **Feature creation** — `total_rooms`, `area_per_bedroom`, `area_per_story`, `bath_per_bed`, `amenity_score`, `area_x_ac`, `area_x_prefarea`, `stories_x_bath`, `area_bin`, and binary flags.
6. **Train/test split** — done before any step that learns from the data, to avoid leakage.
7. **Outliers, skew, scaling** — IQR capping + Yeo-Johnson on skewed columns, standard scaling elsewhere; all fitted on the training set only.
8. **Feature selection** — correlation, VIF, variance threshold, mutual information, RFE, Lasso, random forest importance, PCA; features kept by majority vote across methods.
9. **Validation** — 5-fold cross-validation comparing baseline vs. all-engineered vs. selected features across Ridge, Random Forest, and Gradient Boosting; final check on the held-out test set.

## Results

| Feature set | Model | CV R² |
|---|---|---|
| Baseline (12 original columns) | Ridge | 0.668 |
| Selected (12 features, voted) | Ridge | 0.662 |
| All engineered (23 columns) | Ridge | 0.652 |

Best test-set performance: R² ≈ 0.65, RMSE ≈ ₹1.33M.

The engineered features gave only a marginal change over the original columns on this dataset — most of the gain came from outlier capping, skew correction, and scaling rather than the new features themselves. Documented as an honest finding rather than forcing an improvement.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Author

[Your name here]
