import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Đường dẫn dữ liệu từ chính file này
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Colab_Data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))

def train_decision_tree():
    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ Không tìm thấy file CSV trong thư mục Colab_Data.")
        return None, None

    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(BASE_DIR, f)), reverse=True)
    latest_file = os.path.join(BASE_DIR, csv_files[0])
    print(f"✅ Đọc dữ liệu từ file mới nhất: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # Sử dụng hàm train_test_split để train model, ngoài ra dùng vectorizer là TF-IDF, cây có độ sâu là 10
    X = df['Review']
    y = df['Label']

    vectorizer = TfidfVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.3, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Sau đó dùng model train để dự đoán thử tập test
    y_pred = model.predict(X_test)


    # Trực quan hóa phần ảnh và dự đoán
    visualizations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result", "visualizations"))
    os.makedirs(visualizations_dir, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title("Decision Tree - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(visualizations_dir, "decision_tree_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Classification report (save as text and image)
    report_text = classification_report(y_test, y_pred, digits=4)
    report_txt_path = os.path.join(visualizations_dir, "decision_tree_classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.title("Decision Tree - Classification Report", pad=20)
    plt.text(0.01, 0.05, report_text, family="monospace", fontsize=8)
    report_img_path = os.path.join(visualizations_dir, "decision_tree_classification_report.png")
    plt.tight_layout()
    plt.savefig(report_img_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Lưu lại phần model bằng joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_filename = os.path.join(MODEL_DIR, "merged_DT_model.pkl")
    vectorizer_filename = os.path.join(MODEL_DIR, "merged_DT_vectorizer.pkl")
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    print(f"✅ Đã lưu Decision Tree model và vectorizer vào: {model_filename}, {vectorizer_filename}")
    print(f"📈 Đã lưu Confusion Matrix: {cm_path}")
    print(f"📄 Đã lưu Classification Report: {report_txt_path} và {report_img_path}")
    return model, vectorizer


if __name__ == "__main__":
    train_decision_tree()
