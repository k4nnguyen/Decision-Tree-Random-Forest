import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from saved_model import load_saved_model
from train_decision_tree import train_decision_tree
from train_random_forest import train_random_forest
from crawl import crawl_reviews, chuanHoa
from crawl_batch import crawl_batch
from merge_csvf import merge_all_csv


def veBieuDo(df, model_label):
    # Đảm bảo tồn tại cả 2 cột cho model cụ thể
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

    # 1. Chuẩn bị dữ liệu chung
    # crawl_batch()
    merged_path = merge_all_csv()
    train_decision_tree()
    train_random_forest()

    # 2. Load 2 mô hình và vectorizer
    model_DT, vectorizer_DT = load_saved_model(0)
    model_RF, vectorizer_RF = load_saved_model(1)
    if model_DT is None or vectorizer_DT is None or model_RF is None or vectorizer_RF is None:
        print("❌ Không thể tải đầy đủ mô hình. Dừng lại.")
        return

    # 3. Crawl review cho cửa hàng cần dự đoán
    crawl_reviews(predict_store)
    filename = chuanHoa(predict_store) + ".csv"
    filepath = os.path.join("data", filename)
    df = pd.read_csv(filepath)

    # 4. Tiền xử lý & dự đoán
    reviews = df['Review']
    vec_DT = vectorizer_DT.transform(reviews)
    vec_RF = vectorizer_RF.transform(reviews)
    df['Predict_DT'] = model_DT.predict(vec_DT)
    df['Predict_RF'] = model_RF.predict(vec_RF)

    # 5. Lưu kết quả
    os.makedirs("result", exist_ok=True)
    output_filepath = os.path.join("result", f"classified_{filename}")
    df.to_csv(output_filepath, index=False, encoding="utf-8")

    # 6. Đánh giá & vẽ biểu đồ cho từng mô hình
    for col in ['Predict_DT', 'Predict_RF']:
        tinhGiaTri(df, col)
        veBieuDo(df, col)

    print(f"💾 Đã lưu kết quả vào '{output_filepath}'")

if __name__ == "__main__":
    main()