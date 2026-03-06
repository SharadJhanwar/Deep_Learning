"""
Handling Imbalanced Data
"""
"""
Various Way to Handle Imbalanced Data:
1. Under sampling majority class
2. Over sampling minority class (just duplicating)
3. Over sampling minority class (using SMOTE)
  - Generate synthetic examples using kNN algo
  - SMOTE - Synthetic Minority Over-sampling Technique
4. Ensemble Method 
5. Focal Loss
  - Explanation of focal loss
  - Focal loss will penalize the majority samples during loss calculation and give more weight to minority class in samples

Ex -> Device failure prediction, Cancer prediction, customer churn prediction



2. COPY SAMPLE
If your DataFrame df has 1000 rows and you want 2000 rows by sampling with replacement:

df_2000 = df.sample(n=2000, replace=True, random_state=42)

3. SMOTE , kNN
it uses imblearn

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X,y)


4. ENSEMBLE


"""