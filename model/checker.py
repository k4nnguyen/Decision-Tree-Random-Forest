import joblib
import numpy as np

# Load mô hình và vectorizer
model = joblib.load('merged1_model.pkl')
vectorizer = joblib.load('merged1_vectorizer.pkl')

# Lấy danh sách các từ (features) từ vectorizer
feature_names = vectorizer.get_feature_names_out()

# Kiểm tra xem model có thuộc tính feature_importances_ không
if hasattr(model, 'feature_importances_'):
    # Lấy chỉ số của top 20 từ có độ quan trọng cao nhất
    top_indices = np.argsort(model.feature_importances_)[-20:][::-1]

    print("Top 20 từ quan trọng nhất:")
    cnt = 1
    for idx in top_indices:
        print(f"{cnt}, {feature_names[idx]}: {model.feature_importances_[idx]:.4f}")
        cnt+=1