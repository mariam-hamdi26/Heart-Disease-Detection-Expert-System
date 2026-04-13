import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("D:/collage subjects/semster (6)/Intelligent programming/heart.csv")
print("✅ Dataset loaded — Shape:", df.shape)


for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   Filled '{col}' missing values with median: {median_val}")

print("✅ Missing values handled — Remaining:", df.isnull().sum().sum())


numerical_cols_to_scale = ['trestbps', 'chol', 'thalach', 'oldpeak']
scaler = MinMaxScaler()
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])
print("✅ Normalization done — Columns scaled:", numerical_cols_to_scale)


categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)


bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print("✅ One-Hot Encoding done — Shape after encoding:", df.shape)


target_corr = df.corr()['target'].drop('target').abs().sort_values(ascending=False)


important_features = set(target_corr[target_corr >= 0.1].index.tolist())

important_features.add('chol')


onehot_groups = {
    'sex':     [c for c in df.columns if c.startswith('sex_')],
    'cp':      [c for c in df.columns if c.startswith('cp_')],
    'restecg': [c for c in df.columns if c.startswith('restecg_')],
    'exang':   [c for c in df.columns if c.startswith('exang_')],
    'slope':   [c for c in df.columns if c.startswith('slope_')],
    'ca':      [c for c in df.columns if c.startswith('ca_')],
    'thal':    [c for c in df.columns if c.startswith('thal_')],
}

for group, cols in onehot_groups.items():
    if any(c in important_features for c in cols):
        for c in cols:
            important_features.add(c)

df = df[list(important_features) + ['target']].copy()
print("✅ Feature Selection done — Selected features:", len(important_features))


rename_map = {
    'age':          'Age',
    'trestbps':     'Resting_Blood_Pressure',
    'chol':         'Cholesterol',
    'thalach':      'Maximum_Heart_Rate',
    'oldpeak':      'ST_Depression',
    'sex_0':        'Sex_Female',
    'sex_1':        'Sex_Male',
    'cp_0':         'Chest_Pain_Typical_Angina',
    'cp_1':         'Chest_Pain_Atypical_Angina',
    'cp_2':         'Chest_Pain_Non_Anginal',
    'cp_3':         'Chest_Pain_Asymptomatic',
    'restecg_0.0':  'ECG_Normal',
    'restecg_1.0':  'ECG_ST_Wave_Abnormality',
    'restecg_2.0':  'ECG_Left_Ventricular_Hypertrophy',
    'exang_0':      'Exercise_Angina_No',
    'exang_1':      'Exercise_Angina_Yes',
    'slope_0':      'ST_Slope_Upsloping',
    'slope_1':      'ST_Slope_Flat',
    'slope_2':      'ST_Slope_Downsloping',
    'ca_0':         'Major_Vessels_Count_0',
    'ca_1':         'Major_Vessels_Count_1',
    'ca_2':         'Major_Vessels_Count_2',
    'ca_3':         'Major_Vessels_Count_3',
    'ca_4':         'Major_Vessels_Count_4',
    'thal_0':       'Thalassemia_Normal',
    'thal_1':       'Thalassemia_Fixed_Defect_1',
    'thal_2':       'Thalassemia_Fixed_Defect_2',
    'thal_3':       'Thalassemia_Reversible_Defect',
    'target':       'Heart_Disease_Target',
}
df = df.rename(columns=rename_map)


ordered_num = ['Age', 'Resting_Blood_Pressure', 'Cholesterol',
               'Maximum_Heart_Rate', 'ST_Depression']
ordered_num = [c for c in ordered_num if c in df.columns]

group_order = [
    sorted([c for c in df.columns if c.startswith('Sex_')]),
    sorted([c for c in df.columns if c.startswith('Chest_Pain_')]),
    sorted([c for c in df.columns if c.startswith('ECG_')]),
    sorted([c for c in df.columns if c.startswith('Exercise_')]),
    sorted([c for c in df.columns if c.startswith('ST_Slope_')]),
    sorted([c for c in df.columns if c.startswith('Major_')]),
    sorted([c for c in df.columns if c.startswith('Thalassemia_')]),
]

final_order = ordered_num
for g in group_order:
    final_order += g
final_order += ['Heart_Disease_Target']
df = df[final_order]


df.to_csv("cleaned_data.csv", index=False)
print("✅ Saved: cleaned_data.csv")
print("\n── Final Dataset ─────────────────────────")
print("   Shape  :", df.shape)
print("   Columns:", list(df.columns))
