import time
import pandas as pd
import os
import re
import unicodedata
import ftfy
import nltk
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from webdriver_manager.chrome import ChromeDriverManager
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config.keywords import tichCuc,tieuCuc

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def chuanHoa(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'\s+', '_', text).strip()

def chuanHoaEncoding(text):
    for encoding in ['latin1', 'cp1252']:
        try:
            text = text.encode(encoding).decode('utf-8')
            break
        except:
            continue
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chuanHoaReview(raw_text):
    text = chuanHoaEncoding(raw_text.strip()).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # loại bỏ số và ký tự đặc biệt
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def labelWord(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 1
    elif compound <= -0.05:
        return 0

    # fallback lexicon nếu VADER trung tính
    tokens = text.split()  
    pos_count = sum(1 for w in tokens if w in tichCuc)
    neg_count = sum(1 for w in tokens if w in tieuCuc)

    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return 0
    else:
        return -1

def crawl_reviews(search_query):
    output_folder = "../data/raw_data"
    os.makedirs(output_folder, exist_ok=True)

    # Cấu hình cho Selenium
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) 
    search_url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"
    driver.get(search_url)

    # Tìm tab Reviews
    try:
        review_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@role="tab" and contains(@aria-label, "Reviews")]'))
        )
        review_tab.click()
        print("✅ Đã click vào tab Reviews.")
    except:
        print("❌ Không tìm thấy tab Reviews.")
        driver.quit()
        return

    # Lấy khung cuộn review
    try:
        scrollable_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"))
        )
    except:
        print("❌ Không tìm thấy khung cuộn review chính.")
        driver.quit()
        return

    # Cuộn xuống để load thêm review
    last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
    start_time = time.time()
    max_wait = 60

    for i in range(100):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
        time.sleep(1.5)
        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        print(f"🔄 Cuộn {i+1}, chiều cao: {new_height}")

        if new_height == last_height:
            print("⏹ Không còn review mới để tải.")
            break
        if time.time() - start_time > max_wait:
            print("⏹ Hết thời gian cuộn.")
            break

        last_height = new_height

    print("✅ Đã cuộn xong, bắt đầu thu thập review.")

    # Lấy review block
    review_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.jftiEf')
    print(f"🔎 Số block review tìm thấy: {len(review_blocks)}")

    data = []
    for idx, block in enumerate(review_blocks, start=1):
        try:
            if block.find_elements(By.CSS_SELECTOR, 'div.GvZfFd > div.Jtu6Td'):
                print(f"⏭ Bỏ qua block {idx} (reply của owner).")
                continue

            review_text_elem = block.find_element(By.CLASS_NAME, 'wiI7pd')
            raw_text = review_text_elem.get_attribute("textContent").strip()
            print(f"[{idx}] Raw review:", raw_text)

            processed_text = chuanHoaReview(raw_text)
            print(f"[{idx}] Processed:", processed_text)

            label = labelWord(processed_text)
            print(f"[{idx}] Label:", label)

            if label != -1:
                data.append({"Review": processed_text, "Label": label})
        except Exception as e:
            print(f"⚠️ Lỗi ở block {idx}: {e}")
            continue

    driver.quit()

    # Xuất ra file CSV
    filename = chuanHoa(search_query) + ".csv"
    filepath = os.path.join(output_folder, filename)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"💾 Đã lưu {len(df)} review có nhãn vào '{filepath}'")
