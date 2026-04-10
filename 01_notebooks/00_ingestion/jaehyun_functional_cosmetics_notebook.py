# %%
pip install requests pandas
# %%
import requests
import pandas as pd
import os

# ─────────────────────────────────────────
# 설정 해당 API 및 END_POINT는 활용시 기입
# ─────────────────────────────────────────
# API_KEY   = "SKIN_API"
# END_POINT = "END_POINT"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(BASE_DIR, "기능성화장품_보고품목정보.csv")

# ─────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────
all_items = []
page = 1

while True:
    url = (
        f"{END_POINT}"
        f"?serviceKey={API_KEY}"
        f"&type=json"
        f"&numOfRows=100"
        f"&pageNo={page}"
    )

    response = requests.get(url, timeout=10)
    print(f"페이지 {page} - 상태코드: {response.status_code}")

    if response.status_code != 200:
        print("오류 응답:", response.text[:300])
        break

    data = response.json()

    if page == 1:
        print("응답 키:", list(data.keys()))

    body  = data.get("body", {})
    items = body.get("items", [])

    if not items:
        print("데이터 없음. 종료.")
        break

    all_items.extend(items)

    total_count = body.get("totalCount", 0)
    print(f"  → {len(items)}개 수집 (누적: {len(all_items)} / 전체: {total_count})")

    if len(all_items) >= total_count:
        print("전체 수집 완료!")
        break

    page += 1

# ─────────────────────────────────────────
# CSV 저장
# ─────────────────────────────────────────
df = pd.DataFrame(all_items)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n완료! 총 {len(df)}행 × {len(df.columns)}열")
print(f"저장 위치: {OUTPUT_CSV}")