import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load mô hình và vectorizer
model = joblib.load('../models/model/merged_DT_model.pkl')
vectorizer = joblib.load('../models/model/merged_DT_vectorizer.pkl')

# Lấy tên từ và importance
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_
features = vectorizer.get_feature_names_out()

# Kiểm tra xem model có thuộc tính feature_importances_ không
if hasattr(model, 'feature_importances_'):
    # Lấy chỉ số của top 20 từ có độ quan trọng cao nhất
    top_idx = np.argsort(model.feature_importances_)[-20:][::-1]
    top_features = [features[i] for i in top_idx]
    top_importances = [importances[i] for i in top_idx]
    print("Top 20 từ quan trọng nhất:")
    cnt = 1
    for idx in top_idx:
        print(f"{cnt}, {feature_names[idx]}: {model.feature_importances_[idx]:.4f}")
        cnt+=1

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 từ quan trọng nhất trong Decision Tree")
plt.tight_layout()
plt.show()        