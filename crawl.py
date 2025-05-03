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

# Tải dữ liệu NLP
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Từ khóa để gán nhãn sentiment (có thể mở rộng thêm)
tichCuc = ['good', 'delicious', 'amazing', 'great', 'excellent', 'tasty', 'fresh', 'friendly', 'fast', 'clean', 'cozy', 'affordable', 'perfect', 'hot', 'crispy', 'juicy', 'tender', 'savory', 'yummy', 'pleasant', 'lovely', 'fantastic', 'outstanding', 'flavorful', 'welcoming', 'nice', 'awesome', 'top-notch', 'satisfying', 'polite', 'attentive', 'neat', 'beautiful', 'charming', 'quick', 'well-seasoned', 'generous', 'authentic', 'enjoyable', 'comfortable', 'spotless', 'organized', 'smooth', 'balanced', 'succulent', 'gracious', 'reliable', 'professional', 'hearty', 'mouthwatering']
tieuCuc = ['bad', 'terrible', 'awful', 'disgusting', 'dirty', 'slow', 'rude', 'cold', 'overpriced', 'burnt', 'bland', 'greasy', 'stale', 'soggy', 'salty', 'undercooked', 'raw', 'noisy', 'crowded', 'unhygienic', 'unfriendly', 'forgot', 'waited', 'tiny', 'dry', 'tasteless', 'hard', 'old', 'poor', 'worst', 'expensive', 'inedible', 'horrible', 'annoying', 'smelly', 'disappointing', 'late', 'unacceptable', 'broken', 'tough', 'rubbery', 'oily', 'messy', 'cramped', 'ignored', 'slowly', 'uncomfortable', 'confusing', 'rushed', 'flavorless']

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
    text = text.lower()
    if any(word in text for word in tichCuc):
        return 1
    elif any(word in text for word in tieuCuc):
        return 0
    else:
        return -1  # không xác định

def crawl_reviews(search_query):
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    options = Options()
    options.add_argument("--start-maximized")
    service = Service(executable_path="G:/Selenium/chromedriver-win64/chromedriver.exe")  # Thay đổi đường dẫn đến chromedriver của bạn
    driver = webdriver.Chrome(service=service, options=options)

    search_url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"
    driver.get(search_url)

    try:
        review_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@role="tab" and contains(@aria-label, "Reviews")]'))
        )
        review_tab.click()
        print("Đã click vào tab Reviews.")
    except:
        print("Không tìm thấy tab Reviews.")

    try:
        scrollable_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"))
        )
    except:
        print("Không tìm thấy khung cuộn review chính.")
        driver.quit()
        exit()

    last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
    start_time = time.time()
    max_wait = 60

    for i in range(100):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
        time.sleep(1.5)
        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        print(f"==> Cuộn vòng {i+1}, chiều cao: {new_height}")

        if new_height == last_height:
            print("Không còn review mới để tải.")
            break

        if time.time() - start_time > max_wait:
            print("Hết thời gian cuộn.")
            break

        last_height = new_height

    print("✅ Đã cuộn xong và sẵn sàng thu thập review.")

    review_blocks = driver.find_elements(By.CSS_SELECTOR, 'div.jJc9Ad')
    data = []

    for block in review_blocks:
        try:
            if block.find_elements(By.CSS_SELECTOR, 'div.GvZfFd > div.Jtu6Td'):
                continue
            review_text_elem = block.find_element(By.CLASS_NAME, 'wiI7pd')
            raw_text = review_text_elem.text.strip()
            processed_text = chuanHoaReview(raw_text)
            label = labelWord(processed_text)
            if label != -1:
                data.append({"Review": processed_text, "Label": label})
        except Exception:
            continue

    driver.quit()

    filename = chuanHoa(search_query) + ".csv"
    filepath = os.path.join(output_folder, filename)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"💾 Đã lưu {len(df)} review có nhãn vào file '{filepath}' thành công!")
