"""
Bank Marketing SVC Model v2 - Optimized for speed + ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings, time
import os
warnings.filterwarnings('ignore')

# Path relatif terhadap lokasi script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t0 = time.time()

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target dist:\n{train['berlangganan_deposito'].value_counts()}")

target = train['berlangganan_deposito']
customer_numbers_test = test['customer_number']

# ============================================================
# 2. Feature Engineering
# ============================================================
print("\nFeature Engineering...")

train_len = len(train)
combined = pd.concat([train.drop('berlangganan_deposito', axis=1), test], axis=0, ignore_index=True)
combined.drop('customer_number', axis=1, inplace=True)

# Binary flags
combined['pcontacted'] = (combined['hari_sejak_kontak_sebelumnya'] != 999).astype(int)
combined['was_success'] = (combined['hasil_kampanye_sebelumnya'] == 'success').astype(int)
combined['was_failure'] = (combined['hasil_kampanye_sebelumnya'] == 'failure').astype(int)

# Contact features
combined['total_contacts'] = combined['jumlah_kontak_kampanye_ini'] + combined['jumlah_kontak_sebelumnya']
combined['contact_rate'] = combined['jumlah_kontak_sebelumnya'] / (combined['total_contacts'] + 1)
combined['log_campaign_contacts'] = np.log1p(combined['jumlah_kontak_kampanye_ini'])

# Cyclical month encoding
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
combined['month_num'] = combined['bulan_kontak_terakhir'].map(month_map)
combined['month_sin'] = np.sin(2 * np.pi * combined['month_num'] / 12)
combined['month_cos'] = np.cos(2 * np.pi * combined['month_num'] / 12)

# Day ordinal
day_map = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
combined['day_num'] = combined['hari_kontak_terakhir'].map(day_map)

# Age group
combined['age_group'] = pd.cut(combined['usia'], bins=[0,25,35,45,55,65,100], labels=[0,1,2,3,4,5]).astype(int)

# Economic interactions
combined['euribor_x_employed'] = combined['suku_bunga_euribor_3bln'] * combined['jumlah_pekerja']
combined['euribor_x_cpi'] = combined['suku_bunga_euribor_3bln'] * combined['indeks_harga_konsumen']
combined['cci_x_cpi'] = combined['indeks_kepercayaan_konsumen'] * combined['indeks_harga_konsumen']

# Encode categoricals
cat_cols = ['pekerjaan', 'status_perkawinan', 'pendidikan', 'gagal_bayar_sebelumnya',
            'pinjaman_rumah', 'pinjaman_pribadi', 'jenis_kontak',
            'hasil_kampanye_sebelumnya', 'pulau']

for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Drop replaced columns
combined.drop(['bulan_kontak_terakhir', 'hari_kontak_terakhir', 'month_num'], axis=1, inplace=True)

print(f"Features: {combined.shape[1]} columns")

# ============================================================
# 3. Split and scale
# ============================================================
X_train = combined.iloc[:train_len].values
X_test = combined.iloc[train_len:].values
y_train = target.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. SVC with CalibratedClassifierCV for fast probability estimation
#    Using out-of-fold predictions for calibration
# ============================================================
print("\nTraining SVC with calibrated probabilities...")
print(f"(This may take a few minutes...)")

# Use LinearSVC-like approach or standard SVC
# For better AUC with speed, we use SVC without probability=True
# then wrap with CalibratedClassifierCV

base_svc = SVC(
    kernel='rbf',
    C=1.5,
    gamma='scale',
    class_weight='balanced',
    random_state=42,
    cache_size=2000,
    max_iter=-1  # no limit
)

# CalibratedClassifierCV uses cross-validation to produce calibrated probabilities
# method='sigmoid' is Platt scaling, 'isotonic' is non-parametric
# cv=3 is faster than the default cv=5 used by SVC(probability=True)
calibrated_svc = CalibratedClassifierCV(
    estimator=base_svc,
    method='sigmoid',
    cv=5,
    n_jobs=-1
)

calibrated_svc.fit(X_train_scaled, y_train)
print(f"Training done in {time.time()-t0:.1f}s")

# ============================================================
# 5. Quick OOF evaluation
# ============================================================
print("\nEvaluating with OOF predictions...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
    fold_model = CalibratedClassifierCV(
        estimator=SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced',
                       random_state=42, cache_size=2000),
        method='sigmoid', cv=3
    )
    fold_model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
    oof_preds[val_idx] = fold_model.predict_proba(X_train_scaled[val_idx])[:, 1]
    fold_auc = roc_auc_score(y_train[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

overall_auc = roc_auc_score(y_train, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.5f}")

# ============================================================
# 6. Predict test set
# ============================================================
print("\nGenerating test predictions...")
y_pred_proba = calibrated_svc.predict_proba(X_test_scaled)[:, 1]

print(f"Prediction stats: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

# ============================================================
# 7. Create submission
# ============================================================
submission = pd.DataFrame({
    'customer_number': customer_numbers_test,
    'berlangganan_deposito': y_pred_proba
})

output_path = os.path.join(OUTPUT_DIR, "submission_svc.csv")
submission.to_csv(output_path, index=False)
print(f"\nSubmission saved to: {output_path}")
print(submission.head(10))
print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE!")
