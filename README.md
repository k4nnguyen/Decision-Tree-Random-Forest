# Sử dụng Decision Tree để phân loại bình luận tích cực / tiêu cực

## Tổng quan dự án

Dự án sử dụng selenium để cào dữ liệu từ Google Map, tiền xử lý dữ liệu, vector hóa để tự huấn luyện model, sau đó sẽ sử dụng model đó để dự đoán các bình luận của các quán ăn khác

## Cách chạy chương trình

### Cài đặt các thư viện cần thiết cho chương trình:

- Clone lại dự án:

```bash
git clone https://github.com/k4nnguyen/Decision-Tree.git
```

- Mở terminal ở thư mục đó và chạy

```bash
pip install -r 'requirements.txt'
py setup_nltk.py
```

## Setup Chromedriver

1. Tải Chromedriver phù hợp hệ điều hành tại: https://sites.google.com/chromium.org/driver/
2. Giải nén và tìm kiếm vị trí của chromedriver.exe, ví dụ ổ của mình là G:/Selenium/chromedriver-win64/chromedriver.exe
3. Vào thư mục crawl.py sửa executable_path="vị trí chromedriver.exe"
Để kiểm tra phiên bản Chromedriver phù hợp (Bằng phiên bản chrome của máy), sử dụng trên thanh tìm kiếm:
```bash
chrome://settings/help
```

### Chạy thử code

1. File list.input sẽ là các quán ăn mà các bạn muốn dùng để train model dựa trên các quán đó.
2. Nếu muốn chạy model sẵn có thì có thể comment phần crawl_batch() trong main.py
3. Chỉ cần chạy file main.py và nhập tên quán để dự đoán dựa trên model mới nhất!
```bash
py main.py
```
