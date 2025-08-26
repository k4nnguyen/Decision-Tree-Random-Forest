import joblib
import os

def load_saved_model(x):
    # BASE_DIR là thư mục gốc của dự án
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(BASE_DIR, "models", "model")

    if x == 0:
        model_filename = os.path.join(MODEL_DIR, "merged_DT_model.pkl")
        vectorizer_filename = os.path.join(MODEL_DIR, "merged_DT_vectorizer.pkl")
    else:
        model_filename = os.path.join(MODEL_DIR, "merged_RF_model.pkl")
        vectorizer_filename = os.path.join(MODEL_DIR, "merged_RF_vectorizer.pkl")

    if not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        print(f"❌ Không tìm thấy mô hình hoặc vectorizer.")
        return None, None

    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)

    print(f"✅ Đã tải mô hình và vectorizer.")
    return model, vectorizer
