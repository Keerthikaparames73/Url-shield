import onnxruntime as ort
import numpy as np
import urllib.parse
import re
import math

# Exact Kotlin feature extraction replica
def calculate_entropy(text):
    if not text:
        return 0.0
    entropy = 0.0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    return float(entropy)

def extract_features(url):
    parsed = urllib.parse.urlparse(url if url.startswith('http') else 'http://' + url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    
    url_length = float(len(url))
    ip_pattern = re.compile(r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])")
    has_ip_address = 1.0 if ip_pattern.search(host) else 0.0
    dot_count = float(url.count('.'))
    https_flag = 1.0 if url.lower().startswith("https://") else 0.0
    url_entropy = calculate_entropy(url)
    tokens = [t for t in re.split(r'[^a-zA-Z0-9]', url) if t]
    token_count = float(len(tokens))
    host_parts = host.split('.')
    subdomain_count = float(len(host_parts) - 2) if len(host_parts) > 2 else 0.0
    query_param_count = float(query.count('&') + 1) if query else 0.0
    tld = host_parts[-1] if host_parts else ""
    tld_length = float(len(tld))
    path_length = float(len(path))
    has_hyphen_in_domain = 1.0 if "-" in host else 0.0
    number_of_digits = float(sum(1 for c in url if c.isdigit()))
    common_tlds = ["com", "org", "net", "edu", "gov", "io", "co"]
    tld_popularity = 1.0 if tld.lower() in common_tlds else 0.0
    suspicious_exts = [".exe", ".php", ".js", ".apk", ".bat", ".cmd", ".sh"]
    suspicious_file_extension = 1.0 if any(url.lower().endswith(ext) for ext in suspicious_exts) else 0.0
    domain_name_length = float(len(host))
    percentage_numeric_chars = (number_of_digits / url_length * 100.0) if url_length > 0 else 0.0
    
    return [url_length, has_ip_address, dot_count, https_flag, url_entropy, token_count, 
            subdomain_count, query_param_count, tld_length, path_length, has_hyphen_in_domain, 
            number_of_digits, tld_popularity, suspicious_file_extension, domain_name_length, 
            percentage_numeric_chars]

# Load ONNX model
onnx_path = r"C:\Users\keert\AndroidStudioProjects\URLShield\app\src\main\assets\xgboost_model.onnx"
sess = ort.InferenceSession(onnx_path)

input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

print(f"ONNX Model Inputs: {input_name}")
print(f"ONNX Model Outputs: {output_names}")

test_urls = ["https://www.google.com", "http://paypal-login-security.verify-user.ru"]

for url in test_urls:
    feat = extract_features(url)
    print(f"\nURL: {url}")
    print(f"Array: {feat}")
    
    input_data = np.array([feat], dtype=np.float32)
    result = sess.run(output_names, {input_name: input_data})
    
    label = result[0]
    probs = result[1]
    print(f"Label Output: {label} (Type: {type(label)}, Dtype: {getattr(label, 'dtype', None)})")
    print(f"Probs Output: {probs} (Type: {type(probs)})")
    if len(probs) > 0:
        p0 = probs[0][0]
        p1 = probs[0][1]
        print(f"prob(0) Malicious: {p0}")
        print(f"prob(1) Safe: {p1}")
