from preprocessing.crawl import crawl_reviews

def crawl_batch():
    # Đọc danh sách nhà hàng từ file list.input
    with open('list.input', 'r', encoding='utf-8') as f:
        restaurant_list = [line.strip() for line in f if line.strip()]

    for restaurant in restaurant_list:
        try:
            print(f"\n🚀 Cào: {restaurant}")
            crawl_reviews(restaurant)  # sẽ lưu ở thư mục ../data/raw_data theo mặc định
        except Exception as e:
            print(f"❌ Lỗi với '{restaurant}': {e}")
