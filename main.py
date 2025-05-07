import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from saved_model import load_saved_model
from train_decision_tree import train_decision_tree
from crawl import crawl_reviews, chuanHoa
from crawl_batch import crawl_batch
from merge_csvf import merge_all_csv

import joblib

def veBieuDo(df):
    # Đảm bảo tồn tại cả 2 cột
    if 'Label' not in df.columns or 'Predicted_Label' not in df.columns:
        print("⚠️ Không thể vẽ biểu đồ vì thiếu nhãn thực hoặc nhãn dự đoán.")
        return

    # Tạo bảng đếm số lượng theo từng lớp
    counts = df.groupby(['Label', 'Predicted_Label']).size().reset_index(name='Count')

    # Vẽ biểu đồ dạng heatmap
    pivot = counts.pivot(index='Label', columns='Predicted_Label', values='Count').fillna(0)
    sns.heatmap(pivot, annot=True, fmt='g', cmap='Blues')

    plt.title("So sánh giữa Label thực và Dự đoán")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label") 
    plt.show()

def tinhGiaTri(df):
    # Kiểm tra cột tồn tại
    if 'Label' not in df.columns or 'Predicted_Label' not in df.columns:
        print("⚠️ Thiếu cột 'Label' hoặc 'Predicted_Label'.")
        return

    y_true = df['Label']
    y_pred = df['Predicted_Label']

    # Accuracy, Precision, Recall, F1
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # In ra bảng báo cáo classificationa
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print("\n✅ Đánh giá tổng thể:")
    print(f"- Accuracy: {accuracy:.3f}")
    print(f"- Precision: {precision:.3f}")
    print(f"- Recall: {recall:.3f}")
    print(f"- F1 Score: {f1:.3f}")

def main():
    predict_store = input("Nhập tên cửa hàng bạn muốn dự đoán bình luận (Ví dụ: KFC Hoàng Quốc Việt): ").strip()
    #crawl_batch()
    merged_path = merge_all_csv()
    train_decision_tree()
    # Bước 4: Tải mô hình đã huấn luyện từ cửa hàng
    model, vectorizer = load_saved_model()
    if model is None or vectorizer is None:
        print("❌ Không thể tải mô hình. Quá trình dừng lại.")
        return
    #Bước 5: Cào dữ liệu của cửa hàng muốn dự đoán
    crawl_reviews(predict_store)
    # Đọc dữ liệu từ file CSV của cửa hàng
    filename = chuanHoa(predict_store) + ".csv"
    filepath = os.path.join("data", filename)
    df = pd.read_csv(filepath)

    # Bước 5: Tiền xử lý dữ liệu và phân loại sentiment cho từng review
    reviews = df['Review']
    reviews_vectorized = vectorizer.transform(reviews)
    predictions = model.predict(reviews_vectorized)

    # Bước 6: Ghi lại kết quả phân loại vào file CSV
    df['Predicted_Label'] = predictions
    output_filepath = os.path.join("result", f"classified_{filename}")
    df.to_csv(output_filepath, index=False, encoding="utf-8")
    tinhGiaTri(df)
    veBieuDo(df)
    print(f"💾 Đã phân loại sentiment và lưu vào file '{output_filepath}' thành công!")

if __name__ == "__main__":
    main()
