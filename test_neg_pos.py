import joblib
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load dữ liệu
df = pd.read_csv("data/Colab_Data/merged.csv")
reviews = df['Review'].astype(str).tolist()
y = df['Label'].values

# 2. Fit TF-IDF lên toàn bộ corpus
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(reviews)

# 3. Lấy tên feature
features = vectorizer.get_feature_names_out()

# 4. Tách theo lớp
X_neg = X_vec[y == 0]
X_pos = X_vec[y == 1]

# 5. Tính TF-IDF trung bình mỗi lớp
mean_neg = np.array(X_neg.mean(axis=0)).ravel()
mean_pos = np.array(X_pos.mean(axis=0)).ravel()

# 6. Tính chênh lệch để tìm từ “tích cực” vs “tiêu cực”
delta = mean_pos - mean_neg

# Top 20 từ “tích cực”
top_pos_idx = np.argsort(delta)[::-1][:20]
top_pos = [(features[i], float(delta[i])) for i in top_pos_idx]

# Top 20 từ “tiêu cực”
top_neg_idx = np.argsort(delta)[:20]
top_neg = [(features[i], float(-delta[i])) for i in top_neg_idx]  # lấy độ lớn

# 7. In kết quả
print("Top 20 từ tích cực (xuất hiện nhiều hơn ở lớp positive):")
print(pd.DataFrame(top_pos, columns=["feature", "pos_minus_neg"]))

print("\nTop 20 từ tiêu cực (xuất hiện nhiều hơn ở lớp negative):")
print(pd.DataFrame(top_neg, columns=["feature", "neg_minus_pos"]))
