import pandas as pd
import numpy as np
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import os
import urllib.parse
from sklearn.utils import resample

# 1. EXACT 6-Feature Extraction Logic
def extract_features(url):
    safe_url = url if url.startswith('http') else 'http://' + url
    parsed = urllib.parse.urlparse(safe_url)
    host = parsed.hostname or ""
    
    # 1. url_length
    url_length = float(len(url))
    
    # 2. dot_count
    dot_count = float(url.count('.'))
    
    # 3. special_chars_count (@, -, _)
    special_chars_count = float(url.count('@') + url.count('-') + url.count('_'))
    
    # 4. https_flag
    https_flag = 1.0 if url.lower().startswith("https://") else 0.0
    
    # 5. suspicious_keywords (login, verify, bank, secure, update, account, signin)
    keywords = ["login", "verify", "bank", "secure", "update", "account", "signin"]
    url_lower = url.lower()
    suspicious_keywords = float(sum(1 for kw in keywords if kw in url_lower))
    
    # 6. domain_length
    domain_length = float(len(host))
    
    return [url_length, dot_count, special_chars_count, https_flag, suspicious_keywords, domain_length]


# 2. Load Dataset & Clean
dataset_path = r"C:\Users\keert\Downloads\url_features_extracted.csv"
print(f"Loading data from {dataset_path}...")
df = pd.read_csv(dataset_path)

df = df.dropna(subset=['URL', 'ClassLabel'])

# 3. Label Mapping Explicit Control
# CSV Dataset: 1 = Safe, 0 = Malicious
# WE ENFORCE NEW MAP: Safe = 0, Malicious = 1
def map_label(original_label):
    if original_label == 1:
        return 0 # Safe
    else:
        return 1 # Malicious

df['TargetLabel'] = df['ClassLabel'].apply(map_label)

# 4. Balance Dataset
df_safe = df[df['TargetLabel'] == 0]
df_malicious = df[df['TargetLabel'] == 1]

if len(df_safe) > len(df_malicious):
    df_safe_downsampled = resample(df_safe, replace=False, n_samples=len(df_malicious), random_state=42)
    df_balanced = pd.concat([df_safe_downsampled, df_malicious])
else:
    df_malicious_downsampled = resample(df_malicious, replace=False, n_samples=len(df_safe), random_state=42)
    df_balanced = pd.concat([df_safe, df_malicious_downsampled])

print(f"Balanced Dataset Shape: {df_balanced.shape} | Safe (0): {len(df_balanced[df_balanced['TargetLabel']==0])}, Malicious (1): {len(df_balanced[df_balanced['TargetLabel']==1])}")

# 5. Math Parity Feature Extract
print("Recomputing exact 6 features from raw URL column dynamically...")
X_list = df_balanced['URL'].apply(extract_features).tolist()
X = np.array(X_list, dtype=np.float32)

if np.isnan(X).any():
    X = np.nan_to_num(X)

y = df_balanced['TargetLabel'].values.astype(int)

# 6. Model Training
print("Training SIMPLIFIED XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.4f}")

# 7. Verification Proof
print("\n--- MODEL MANUAL VERIFICATION ---")
test_urls = ["https://www.google.com", "http://paypal-login-security.verify-user.ru"]
for t_url in test_urls:
    t_feat = np.array([extract_features(t_url)], dtype=np.float32)
    t_probs = model.predict_proba(t_feat)[0]
    prob_safe = t_probs[0] # index 0 is mapped to class 0 (Safe)
    prob_malicious = t_probs[1] # index 1 is mapped to class 1 (Malicious)
    
    result = "SAFE" if prob_safe > prob_malicious else "MALICIOUS"
    
    print(f"URL: {t_url}")
    print(f" > Features: {t_feat[0]}")
    print(f" > Pred Array: [Safe(0): {prob_safe:.4f}, Malicious(1): {prob_malicious:.4f}] => {result}")

# 8. Export ONNX (Input Shape: [None, 6])
print("\nConverting to ONNX format...")
initial_types = [('float_input', FloatTensorType([None, 6]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=14)

app_assets_dir = r"C:\Users\keert\AndroidStudioProjects\URLShield\app\src\main\assets"
os.makedirs(app_assets_dir, exist_ok=True)
onnx_path = os.path.join(app_assets_dir, "xgboost_model.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model successfully saved to {onnx_path} (Ready for Android)")
