# pip install requests beautifulsoup4 openpyxl pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote
import time
import os

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))          # 01_notebooks/00_ingestion/
RAW_DIR     = os.path.join(BASE_DIR, "..", "..", "00_data", "00_raw")  # 00_data/00_raw/
RAW_DIR     = os.path.abspath(RAW_DIR)                            # 절대경로로 정리

INPUT_CSV   = os.path.join(RAW_DIR, "화장품원료성분정보조회_20260406.csv")
OUTPUT_XLSX = os.path.join(RAW_DIR, "coos_성분정보.xlsx")

print(f"입력 경로: {INPUT_CSV}")
print(f"출력 경로: {OUTPUT_XLSX}")

# ─────────────────────────────────────────
# CSV 불러오기
# ─────────────────────────────────────────
df_input = pd.read_csv(INPUT_CSV, encoding="cp949")
ingredient_list = df_input["표준명 [INGR_KOR_NAME] "].dropna().tolist()

print(f"총 {len(ingredient_list)}개 성분 크롤링 시작")

results = []

# ─────────────────────────────────────────
# 크롤링 루프
# ─────────────────────────────────────────
for i, name in enumerate(ingredient_list):
    name = name.strip()
    url = f"https://coos.kr/ingredients/{quote(name)}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        data = {"성분명": name, "URL": url}

        # 테이블 데이터 파싱
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all(["th", "td"])
                if len(cols) == 2:
                    key   = cols[0].get_text(strip=True)
                    value = cols[1].get_text(separator=" ", strip=True)
                    data[key] = value

        # AI 설명 추출
        ai_section = soup.find("div", string=lambda t: t and "AI Description" in t)
        if ai_section:
            ai_p = ai_section.find_next("p")
            data["AI설명"] = ai_p.get_text(strip=True) if ai_p else ""
        else:
            ai_desc = soup.find("p")
            data["AI설명"] = ai_desc.get_text(strip=True) if ai_desc else ""

        results.append(data)

    except Exception as e:
        print(f"[오류] {name} → {e}")
        continue

    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(ingredient_list)} 완료...")

    time.sleep(0.5)

# ─────────────────────────────────────────
# 엑셀 저장
# ─────────────────────────────────────────
df_output = pd.DataFrame(results)
df_output.to_excel(OUTPUT_XLSX, index=False)
print(f"\n완료! 총 {len(results)}개 성분 저장됨 → {OUTPUT_XLSX}")