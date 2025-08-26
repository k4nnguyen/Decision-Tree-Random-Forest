import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.saved_model import load_saved_model
from models.train_decision_tree import train_decision_tree
from models.train_random_forest import train_random_forest
from preprocessing.crawl import crawl_reviews, chuanHoa
from preprocessing.crawl_batch import crawl_batch
from preprocessing.merge_csvf import merge_all_csv

# Lấy đường dẫn gốc dự án
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_data")
RESULT_DATA_DIR = os.path.join(BASE_DIR, "result", "data")


def veBieuDo(df, model_label):
    true_col = 'Label'
    pred_col = model_label
    if true_col not in df.columns or pred_col not in df.columns:
        print(f"⚠️ Không thể vẽ biểu đồ cho {model_label} vì thiếu cột.")
        return

    counts = df.groupby([true_col, pred_col]).size().reset_index(name='Count')
    pivot = counts.pivot(index=true_col, columns=pred_col, values='Count').fillna(0)
    sns.heatmap(pivot, annot=True, fmt='g', cmap='Blues')
    plt.title(f"True vs Predicted ({model_label})")
    plt.xlabel(f"Predicted ({model_label})")
    plt.ylabel("True Label")
    plt.show()
    return pivot


def tinhGiaTri(df, model_label):
    true_col = 'Label'
    pred_col = model_label
    if true_col not in df.columns or pred_col not in df.columns:
        print(f"⚠️ Thiếu cột cho {model_label}.")
        return

    y_true = df[true_col]
    y_pred = df[pred_col]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n📊 Classification Report ({model_label}):")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"\n✅ Đánh giá tổng thể ({model_label}):")
    print(f"- Accuracy: {accuracy:.3f}")
    print(f"- Precision: {precision:.3f}")
    print(f"- Recall: {recall:.3f}")
    print(f"- F1 Score: {f1:.3f}")

    return accuracy, precision, recall, f1


def main():
    predict_store = input("Nhập tên cửa hàng bạn muốn dự đoán bình luận (Ví dụ: KFC Hoàng Quốc Việt): ").strip()

    # 1. Hỏi người dùng có crawl dữ liệu mới không
    while True:
        decision = input("Bạn có muốn tìm thêm dữ liệu mới trong danh sách không? (Y/N): ").strip().upper()
        if decision == "Y":
            print("Đang thực hiện crawl dữ liệu mới...")
            crawl_batch()
            break
        elif decision == "N":
            print("Bỏ qua bước crawl dữ liệu.")
            break
        else:
            print("❌ Vui lòng nhập Y hoặc N.")

    # 2. Merge dữ liệu nếu có thư mục raw_data
    if not os.path.exists(RAW_DATA_DIR) or len(os.listdir(RAW_DATA_DIR)) == 0:
        print("⚠️ Không có dữ liệu trong raw_data. Bỏ qua bước merge và train.")
        return

    merged_path = merge_all_csv()
    train_decision_tree()
    train_random_forest()

    # 3. Load 2 mô hình và vectorizer
    model_DT, vectorizer_DT = load_saved_model(0)
    model_RF, vectorizer_RF = load_saved_model(1)
    if model_DT is None or vectorizer_DT is None or model_RF is None or vectorizer_RF is None:
        print("❌ Không thể tải đầy đủ mô hình. Dừng lại.")
        return

    # 4. Crawl review cho cửa hàng cần dự đoán
    crawl_reviews(predict_store)
    filename = chuanHoa(predict_store) + ".csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # 5. Tiền xử lý & dự đoán
    reviews = df['Review']
    vec_DT = vectorizer_DT.transform(reviews)
    vec_RF = vectorizer_RF.transform(reviews)
    df['Predict_DT'] = model_DT.predict(vec_DT)
    df['Predict_RF'] = model_RF.predict(vec_RF)

    # 6. Lưu kết quả vào ../result/data
    os.makedirs(RESULT_DATA_DIR, exist_ok=True)
    output_filepath = os.path.join(RESULT_DATA_DIR, f"classified_{filename}")
    df.to_csv(output_filepath, index=False, encoding="utf-8")

    # 7. Đánh giá & vẽ biểu đồ
    for col in ['Predict_DT', 'Predict_RF']:
        tinhGiaTri(df, col)
        veBieuDo(df, col)

    print(f"💾 Đã lưu kết quả vào '{output_filepath}'")


if __name__ == "__main__":
    main()
