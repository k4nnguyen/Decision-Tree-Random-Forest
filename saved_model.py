import joblib
import os

def load_saved_model(x):
    if(x == 0):
        model_filename = os.path.join("model", f"merged_DT_model.pkl")
        vectorizer_filename = os.path.join("model", f"merged_DT_vectorizer.pkl")
    else:
        model_filename = os.path.join("model", f"merged_RF_model.pkl")
        vectorizer_filename = os.path.join("model", f"merged_RF_vectorizer.pkl")

    # Kiểm tra xem mô hình và vectorizer có tồn tại không
    if not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        print(f"❌ Không tìm thấy mô hình hoặc vectorizer cho cửa hàng tổng hợp.")
        return None, None

    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)

    print(f"✅ Đã tải mô hình và vectorizer cho cửa hàng tổng hợp.")
    return model, vectorizer
