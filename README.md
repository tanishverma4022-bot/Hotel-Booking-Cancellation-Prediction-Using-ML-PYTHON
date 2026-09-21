# Hotel-Booking-Cancellation-Prediction-Using-ML-PYTHON
A Python machine learning project that predicts hotel booking cancellations using Logistic Regression and Decision Tree models. Includes data cleaning, exploratory analysis, feature engineering, preprocessing pipelines, cross-validation, hyperparameter tuning and model evaluation.
This project uses hotel booking details to predict cancellations. I check and clean the data,
explore booking patterns, create a few useful features and compare Logistic Regression with
a Decision Tree. The notebook contains the code, charts, observations and final results.
The background notes and references are collected here.

## Dataset

- 119,390 historical bookings and 32 columns from two hotels in Portugal.
- Arrival dates cover July 2015 to August 2017.
- Target: `is_canceled` (0 = not cancelled, 1 = cancelled).
- Original authors: Nuno Antonio, Ana de Almeida and Luis Nunes (2019).
- [Research article](https://doi.org/10.1016/j.dib.2018.11.126).
- [TidyTuesday download and data dictionary](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md).
- Original article/data license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


## Project Steps

1. Load the dataset using `pd.read_csv` with a full Windows path.
2. Check duplicate rows, missing values, data types and numerical summaries.
3. Remove exact duplicates and limit the analysis to overnight bookings with nonzero guest counts.
4. Explore cancellation rates by hotel and market segment, and compare lead times.
5. Create total nights, total guests, previous bookings and previous cancellation rate.
6. Remove outcome-related fields and selected fields with unclear availability at prediction time.
7. Split into 80% training and 20% test data using stratification.
8. Use median/mode imputation, scaling and one-hot encoding within pipelines.
9. Train baseline Logistic Regression; tune balanced Logistic Regression with five-fold GridSearchCV.
10. Compare with a depth-5 Decision Tree using training cross-validation ROC-AUC.
11. Evaluate the selected model once on the test set at the default threshold.
12. Interpret Logistic Regression coefficients and write conclusions.

## Questions Answered by This Project

| Question | Finding in this run |
|---|---|
| What is the cleaned cancellation rate? | 27.68% of 86,639 bookings. |
| Which hotel has the higher rate? | City Hotel: 30.21%; Resort Hotel: 23.71%. |
| How do market segments differ? | Online TA: 35.55%; Direct: 14.87%. The Undefined group has only two records. |
| How does lead time differ? | Cancelled: mean 105.82 days; not cancelled: mean 70.52 days. |
| Which model is selected? | Baseline Logistic Regression; its CV ROC-AUC is almost tied with the tuned model. |
| How many cancellations are identified? | 1,614 of 4,797; 3,183 missed and 988 false alarms. |
| What do the coefficients show? | Country categories have the largest absolute tuned-LR coefficients, but these are not causal or stable policy rules. |

## Model Results

Selected model: **Baseline Logistic Regression**. Test set: **17,328 bookings**.

| Metric | Test result |
|---|---:|
| Accuracy | 75.93% |
| Precision | 62.03% |
| Recall | 33.65% |
| F1 | 0.436 |
| ROC-AUC | 0.779 |
| Average Precision | 0.585 |

An always-not-cancelled prediction would have about 72.3% accuracy but zero cancellation
recall. The model's 33.6% recall means it still misses many cancellations. The nearly
identical baseline/tuned CV scores do not show a meaningful tuning improvement.
Average Precision is reported by its precise name, rather than labelled trapezoidal PR-AUC.

![Test confusion matrix](Results/05_confusion_matrix.png)

  
## Limitations

- Removing 31,994 exact repeated rows is a modeling choice; identical records might be different bookings.
- After deduplication, 757 additional records with zero guests or zero nights are excluded. Missing children values stay missing for pipeline imputation.
- Remaining data is not representative of the original cancellation rate or all hotels.
- Exploratory charts use the cleaned dataset before splitting, as in the class project; this is an educational evaluation.
- A random split does not measure future-period performance. Related bookings/guests may span splits.
- Final status, assigned room, booking amendments and other uncertain fields are excluded, but historical snapshots still cannot verify all inputs existed at booking time.
- The coefficient chart explains the tuned Logistic Regression; the selected final model is the baseline version.
- Country coefficients are associations, not evidence of causation or a basis for nationality-based treatment.
- No app, deployment, calibrated probabilities or measured business savings are included.
- Zero-night entries could be day-use bookings. Their exclusion defines the project scope rather than proving they are errors.
- Missing agent/company IDs can mean no agent/company was involved; they are not treated as missing numerical measurements.
- The dataset covers only two Portuguese hotels in 2015-2017, so results may change for other hotels or later years.
- The highest grid-search CV score was used for selection and can be optimistic.
- Numerical inputs are standardized, but category flags are not. Their coefficient magnitudes are not directly equivalent measures of importance.
- Rare categories and correlated features can affect coefficients. With every category encoded and regularization applied, category coefficients are not simple comparisons against one omitted reference category.

## What I Learned

- How to check a dataset for missing values, duplicate records and unusual entries.
- How to use `groupby`, bar charts and boxplots to compare booking patterns.
- How to create features such as total nights, total guests and previous cancellation rate.
- How to combine imputation, scaling and one-hot encoding in a preprocessing pipeline.
- How to train Logistic Regression and a Decision Tree, and use GridSearchCV for basic tuning.
- Why model selection should use training cross-validation rather than test scores.
- Why accuracy alone is not enough: a model can have reasonable accuracy while missing many cancellations.
- How to read a confusion matrix and explain precision, recall, F1, ROC-AUC and Average Precision.

