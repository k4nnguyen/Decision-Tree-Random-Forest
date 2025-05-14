import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_random_forest():
    # Đường dẫn dữ liệu gộp
    data_dir = "data/Colab_Data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print("❌ Không tìm thấy file CSV trong thư mục Colab_Data.")
        return None, None

    # Chọn file mới nhất
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    latest_file = os.path.join(data_dir, csv_files[0])
    print(f"✅ Đọc dữ liệu từ file mới nhất: {latest_file}")

    df = pd.read_csv(latest_file)
    X = df['Review']
    y = df['Label']

    # Vector hóa TF-IDF giống train_decision_tree
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)
    # TFidf để vector hóa các từ thành các vector số, trọng số -> độ quan trọng từ đó

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.3, random_state=42, stratify=y
    )

    # Khởi tạo và huấn luyện Random Forest
    model = RandomForestClassifier(
        class_weight = 'balanced',
        #cân bằng trọng số (ít mẫu trọng số sẽ cao)
        n_estimators=100,
        #số lượng cây trong 'rừng'
        max_depth=10,
        #độ sâu của cây
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    model_filename = os.path.join("model", "merged_RF_model.pkl")
    vectorizer_filename = os.path.join("model", "merged_RF_vectorizer.pkl")
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

    print(f"✅ Đã lưu Random Forest model và vectorizer vào: {model_filename}, {vectorizer_filename}")
    return model, vectorizer
