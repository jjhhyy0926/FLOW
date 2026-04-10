# pip install selenium webdriver-manager
# pip install requests beautifulsoup4 openpyxl pandas selenium webdriver-manager
# 터미널에 설치

import time
import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "paulaschoice_ingredients.csv")


# 해당 URL을 휼륭함, 좋음, 보통, 나쁨, 매우 나쁨 사이트 주소로 수정하여 사용
BASE_URL = (
    "https://www.paulaschoice.co.kr/ingredients"
    "?csortb1=name&csortd1=1"
    "&crefn1=ingredientRating&crefv1=%ED%9B%8C%EB%A5%AD%ED%95%A8"
    "&sz=10&start={start}"
)

# ─────────────────────────────────────────
# WebDriver 초기화
# ─────────────────────────────────────────
def init_driver():
    options = Options()
    # options.add_argument("--headless")        # 브라우저 창 숨기려면 주석 해제
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


# ─────────────────────────────────────────
# 목록 페이지에서 성분 링크 수집
# ─────────────────────────────────────────
def get_ingredient_links(driver):
    links = driver.find_elements(
        By.CSS_SELECTOR, "a[href*='/ingredients/ingredient-']"
    )
    seen   = set()
    unique = []
    for el in links:
        href = el.get_attribute("href")
        if href and href not in seen:
            seen.add(href)
            unique.append(href)
    return unique


# ─────────────────────────────────────────
# 상세 페이지 스크래핑
# ─────────────────────────────────────────
def scrape_detail_page(driver, detail_url):
    driver.get(detail_url)
    time.sleep(1.5)

    data = {}

    try:
        desc_title = driver.find_element(
            By.XPATH, "//div[contains(@class, 'DescriptionTitle')]"
        )
        data["한글명"] = desc_title.text.replace("성분소개", "").strip()
    except NoSuchElementException:
        data["한글명"] = ""

    try:
        data["영문명"] = driver.find_element(
            By.CSS_SELECTOR, "h3[class*='EnglishName']"
        ).text.strip()
    except NoSuchElementException:
        data["영문명"] = ""

    try:
        data["등급"] = driver.find_element(
            By.CSS_SELECTOR, "span.ColoredIngredientRating__Rating-sc-r02772-0"
        ).text.strip()
    except NoSuchElementException:
        data["등급"] = ""

    try:
        benefits_div = driver.find_element(By.CSS_SELECTOR, "div[class*='Benefits']")
        benefit_links = benefits_div.find_elements(By.TAG_NAME, "a")
        data["효과별"] = ", ".join(a.text.strip() for a in benefit_links if a.text.strip())
    except NoSuchElementException:
        data["효과별"] = ""

    try:
        category_div = driver.find_element(
            By.XPATH, "//div[starts-with(normalize-space(.), '분류:')]"
        )
        data["분류"] = category_div.text.replace("분류:", "").strip()
    except NoSuchElementException:
        data["분류"] = ""

    try:
        desc_body = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'DescriptionTitle')]/following-sibling::div[1]"
        )
        data["성분설명"] = desc_body.text.strip()
    except NoSuchElementException:
        data["성분설명"] = ""

    try:
        related_div = driver.find_element(By.CSS_SELECTOR, "div[class*='RelatedWrapper']")
        related_links = related_div.find_elements(By.TAG_NAME, "a")
        data["연관성분"] = ", ".join(a.text.strip() for a in related_links if a.text.strip())
    except NoSuchElementException:
        data["연관성분"] = ""

    try:
        ref_divs = driver.find_elements(By.CSS_SELECTOR, "div[class*='__Reference-']")
        data["참고논문"] = "\n".join(d.text.strip() for d in ref_divs if d.text.strip())
    except NoSuchElementException:
        data["참고논문"] = ""

    return data


# ─────────────────────────────────────────
# 메인 크롤링 루프
# ─────────────────────────────────────────
FIELDNAMES = ["한글명", "영문명", "등급", "효과별", "분류", "성분설명", "연관성분", "참고논문"]

def main():
    driver = init_driver()
    results = []
    current_start = 0
    page_num = 1

    try:
        while True:
            list_url = BASE_URL.format(start=current_start)
            print(f"\n[페이지 {page_num}] 로딩 중... (start={current_start})")

            driver.get(list_url)
            time.sleep(2)

            ingredient_links = get_ingredient_links(driver)
            print(f"  → 링크 {len(ingredient_links)}개 발견")

            if not ingredient_links:
                print("  → 더 이상 성분 없음. 크롤링 종료.")
                break

            for idx, link in enumerate(ingredient_links):
                print(f"  [{idx+1}/{len(ingredient_links)}] {link.split('/')[-1][:40]}")

                try:
                    detail_data = scrape_detail_page(driver, link)
                    results.append(detail_data)
                except Exception as e:
                    print(f"    !! 수집 실패: {e}")
                    results.append({
                        "한글명": "수집실패", "영문명": link, "등급": "",
                        "효과별": "", "분류": "", "성분설명": "",
                        "연관성분": "", "참고논문": ""
                    })

                driver.get(list_url)
                time.sleep(1.5)

            # 페이지마다 중간 저장
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(results)
            print(f"  → 중간 저장 완료 (누적 {len(results)}개)")

            current_start += 10
            page_num += 1

    finally:
        driver.quit()
        print(f"\n크롤링 완료! 총 {len(results)}개 성분 수집")
        print(f"저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

import csv
import os
import pandas as pd

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "paulaschoice_ingredients.csv")
FIELDNAMES  = ["한글명", "영문명", "등급", "효과별", "분류", "성분설명", "연관성분", "참고논문"]

# ─────────────────────────────────────────
# 최종 CSV 저장
# ─────────────────────────────────────────
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(results)

print(f"최종 저장 완료 → {OUTPUT_FILE} ({len(results)}개)")

# ─────────────────────────────────────────
# 저장 결과 확인
# ─────────────────────────────────────────
df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
print(f"컬럼: {list(df.columns)}")
print(f"행 수: {len(df)}")
print(df.head().to_string())   # 파이참은 df.head() 단독으론 출력 안 됨 → to_string() 사용