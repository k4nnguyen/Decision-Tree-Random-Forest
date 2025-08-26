import os
import pandas as pd

# Đường dẫn tuyệt đối dựa trên vị trí file này
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
COLAB_DATA_DIR = os.path.join(BASE_DIR, "Colab_Data")

def merge_all_csv(
    source_folder=RAW_DATA_DIR, 
    output_folder=COLAB_DATA_DIR, 
    output_filename="merged.csv"
):
    os.makedirs(output_folder, exist_ok=True)
    merged_df = pd.DataFrame()

    for filename in os.listdir(source_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_folder, filename)
            try:
                df = pd.read_csv(file_path)
                if 'Review' in df.columns and 'Label' in df.columns:
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                    print(f"✅ Đã gộp: {filename}")
                else:
                    print(f"⚠️ Bỏ qua file không hợp lệ: {filename}")
            except Exception as e:
                print(f"❌ Lỗi đọc file {filename}: {e}")

    output_path = os.path.join(output_folder, output_filename)
    merged_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n🎉 Đã lưu file gộp tại: {output_path}")
    return output_path
