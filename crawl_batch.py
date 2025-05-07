import os
import shutil
from crawl import crawl_reviews

# Đường dẫn lưu cố định
SAVE_DIR = r'G:/z/Python/BTL_Py/Code/data'
os.makedirs(SAVE_DIR, exist_ok=True)

def crawl_batch():
    # Đọc danh sách nhà hàng từ file list.input
    with open('list.input', 'r', encoding='utf-8') as f:
        restaurant_list = [line.strip() for line in f if line.strip()]

    for restaurant in restaurant_list:
        try:
            print(f"\n🚀 Cào: {restaurant}")
            crawl_reviews(restaurant)  # sẽ lưu ở thư mục ./data theo mặc định

            # Di chuyển file từ ./data sang thư mục SAVE_DIR
            safe_filename = restaurant.replace(" ", "_").replace("/", "_") + ".csv"
            src = os.path.join("data", safe_filename)
            dst = os.path.join(SAVE_DIR, safe_filename)

            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"✅ Đã di chuyển: {dst}")
            else:
                print(f"⚠️ Không tìm thấy file tạm: {src}")
        except Exception as e:
            print(f"❌ Lỗi với '{restaurant}': {e}")
