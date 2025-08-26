# Sentiment Classification using Decision Tree & Random Forest

## 📌 Giới thiệu dự án

Dự án này thực hiện **phân loại cảm xúc của bình luận** (Sentiment Analysis) bằng các thuật toán **Decision Tree** và **Random Forest**. Hệ thống bao gồm các bước: thu thập dữ liệu (cào từ Google Maps), tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá mô hình, và triển khai kết quả phân loại.

## 🎯 Mục tiêu

- Hiểu và triển khai hai thuật toán phổ biến trong Machine Learning: **Decision Tree** và **Random Forest**.
- Thực hành thu thập dữ liệu từ Google Maps bằng: **Selenium**
- Ứng dụng vào bài toán thực tế: **Phân loại bình luận tích cực và tiêu cực**.
- Tạo pipeline xử lý dữ liệu từ thu thập, tiền xử lý đến phân loại và đánh giá.
- Vẽ được biểu đồ để trực quan hóa dữ liệu bằng **Seaborn** và **Matplotlib.pyplot**

## 🏗 Cấu trúc dự án

```
BTL_Py/
│
├── app/
│   └── main.py                # File chạy chính
│
├── crawl/
│   ├── crawl.py               # Cào dữ liệu Google Maps
│   ├── crawl_batch.py         # Cào nhiều địa điểm
│
├── models/
│   ├── train_decision_tree.py # Huấn luyện Decision Tree
│   ├── saved_model.py         # Lưu và tải mô hình
│
├── preprocessing/
│   ├── merge_csvf.py          # Gộp file CSV
│
├── data/
│   ├── raw_data/              # Dữ liệu gốc
│   ├── processed_data/        # Dữ liệu sau tiền xử lý
│
├── result/
│   └── data/                  # Kết quả phân loại
│
└── README.md
```

## 🛠 Công nghệ sử dụng

- **Python 3.10**
- **Selenium** (Thu thập dữ liệu)
- **Pandas, NumPy** (Xử lý dữ liệu)
- **Scikit-learn** (Decision Tree, Random Forest)
- **Joblib** (Lưu và tải mô hình)

## 🚀 Các tính năng chính

- ✅ **Cào dữ liệu đánh giá từ Google Maps**
- ✅ **Tiền xử lý dữ liệu** (Chuẩn hóa, làm sạch, loại bỏ emoji)
- ✅ **Huấn luyện mô hình phân loại sentiment**
- ✅ **Đánh giá và xuất kết quả**
- ✅ **Lưu và tái sử dụng mô hình**

## 🔧 Cài đặt dự án

1. Clone dự án:

   ```bash
   git clone https://github.com/k4nnguyen/Decision-Tree-Random-Forest.git
   cd Decision-Tree-Random-Forest
   ```

2. Tạo môi trường ảo và cài đặt dependencies:

   ```bash

   ```

# Tạo môi trường ảo

python -m venv venv

# Kích hoạt môi trường ảo

# Trên macOS/Linux:

source venv/bin/activate

# Trên Windows:

venv\Scripts\activate

# Cài đặt các thư viện Python cần thiết

pip install -r requirements.txt

# Chạy script setup_nltk.py để tải dữ liệu NLTK

python features/setup_nltk.py

````

3. Chạy ứng dụng:
   ```bash
   cd app
   python main.py
````

## 📚 Tài liệu tham khảo

- [Selenium](https://www.selenium.dev/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Google Maps Review Scraping](https://serpapi.com/maps-local-results)
- [Machine Learning with Python](https://www.machinelearningplus.com)

## 📞 Liên hệ

- **Tác giả:** Nguyễn Kim An
- **Email:** annguyenne2906@gmail.com
- **GitHub:** [k4nnguyen](https://github.com/k4nnguyen)
