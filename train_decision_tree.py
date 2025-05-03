import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(store_name):
    # Đọc dữ liệu
    data_dir = "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print("❌ Không tìm thấy file CSV trong thư mục data.")
        return

    # Chọn file CSV mới nhất
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    latest_file = os.path.join(data_dir, csv_files[0])

    print(f"✅ Đọc dữ liệu từ file mới nhất: {latest_file}")
    df = pd.read_csv(latest_file)

    # Tách dữ liệu và nhãn
    X = df['Review']
    y = df['Label']

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

    # Huấn luyện mô hình Decision Tree
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Tạo thư mục "model" nếu chưa có
    os.makedirs("model", exist_ok=True)

    # Lưu mô hình và vectorizer vào thư mục "model"
    model_filename = os.path.join("model", f"{store_name}_model.pkl")
    vectorizer_filename = os.path.join("model", f"{store_name}_vectorizer.pkl")
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

    print(f"✅ Đã lưu mô hình và vectorizer cho cửa hàng '{store_name}' vào thư mục 'model' với tên {model_filename} và {vectorizer_filename}.")
    return model, vectorizer
