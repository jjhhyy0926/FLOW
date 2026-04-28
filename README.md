<h1 align="center">🫧 DermaLense by Flow</h1>


<p align="center">
 <b>복수의 뷰티 데이터를 통합한 RAG 기반 AI 스킨케어 의사결정 서비스</b><br>
 파편화된 성분 정보를 통합하고, 개인 피부에 맞는 제품을 데이터 기반으로 추천합니다.
</p>


<p align="center">
 <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
 <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
 <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
 <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
 <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" />
 <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
 <a href="https://www.notion.so/ohgiraffers/3RD_PROJECT-33d649136c11806eb17fda8ac18be5f4"><img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white" /></a>
</p>


---

## ✨ Key Features

- 🔍 **병렬 하이브리드 검색** — BM25 + Dense (HyDE) + RRF로 키워드·의미 맥락을 통합 반영
- 🤖 **질문 유형 자동 라우팅** — 성분 분석·제품 추천·일반 질의를 분류해 최적 검색 방식 적용
- ⚖️ **WoE 기반 이중 가중치** — 출처 신뢰도 × 성분 중요도 조합으로 결과 정렬
- 📸 **OCR 성분표 인식** — PaddleOCR로 화장품 패키지 촬영 시 성분 자동 추출·분석

---

## 🚀 Getting Started

> 프로젝트 루트 기준으로 실행합니다.

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env에 아래 키 입력:
# OPENAI_API_KEY=...
# COHERE_API_KEY=...
```

### 3. 데이터 파이프라인

```bash
python 03_scripts/01_validate_raw.py
python 03_scripts/02_make_dataset.py
python 03_scripts/03_build_features.py


# ⚠️ 02_src 폴더를 Mark Directory As → Sources Root 설정


# FAISS 인덱스 생성 (OpenAI API 비용 발생)
python 03_scripts/04_train.py --preset_id 1
python 03_scripts/04_train.py --preset_id 2
python 03_scripts/04_train.py --preset_id 3
python 03_scripts/04_train.py --preset_id 4
```

### 4. 서비스 실행

```bash
# 터미널 1 — FastAPI 백엔드
uvicorn api_server:app --reload


# 터미널 2 — Streamlit 프론트엔드
streamlit run streamlit_app.py
```

> ⚠️ **주의사항**
> - `04_train.py`는 OpenAI 임베딩 API를 호출하므로 비용이 발생합니다
> - PaddleOCR은 첫 실행 시 모델을 자동 다운로드합니다 (수 분 소요)
> - 4, 5단계는 터미널 두 개를 동시에 열어서 실행해야 합니다


---

## 🏗 시스템 아키텍처

### 전체 아키텍처

![Architecture](./06_assets/architecture.png)

### RAG 파이프라인

![RAG Pipeline](./06_assets/rag_pipeline_architecture.png)


---

## 🔬 Technical Deep Dive

<details>
<summary><b>📊 데이터 설계 & ERD</b></summary>

### ERD

![ERD](./06_assets/erd.png)

3개 소스(coos.kr, 화해, Paula's Choice)의 이종 데이터를 **CAS No. → INCI 영문명 → 표준 한글명 → 이명 사전** 순서의 계층적 매핑으로 표준화합니다.

| 테이블                | 설명            | 주요 컬럼                                                      |
|:-------------------|:--------------|:-----------------------------------------------------------|
| `ewg_chunk`        | 성분별 안전 등급·점수  | `ingredient_ko`, `hw_ewg`, `coos_score`, `pc_grade`        |
| `basic_info_chunk` | 기본 기능·카테고리    | `pc_effect`, `pc_category`, `coos_function`, `hw_purpose`  |
| `expert_chunk`     | 전문가 평가·국가별 기준 | `pc_description`, `coos_ai_desc`, `coos_KR`, `hw_category` |
| `product_chunk`    | 화장품 제품 DB     | 제품명, 성분 리스트, 브랜드, 카테고리                                     |

</details>


<details>
<summary><b>⚖️ WoE 기반 가중치 시스템</b></summary>


DermaLens는 3개 데이터 소스(coos.kr, 화해, Paula's Choice)의 이질적인 성분 안전성 등급을 통합하기 위해 **Weight of Evidence(WoE)** 방법론을 적용했습니다.

**핵심 공식:**

$$
W = -\ln\left(\frac{Dist_{Others}}{Dist_{Good}}\right)
$$

- **양수(+):** 해당 등급에 Good 성분이 밀집 → 추천 우선순위 상향
- **0 (중립):** 변별력 없음 → 추천 근거로 부적합
- **음수(−):** 해당 등급에 Others 성분이 밀집 → 추천에서 배제

**전처리 3단계:**

1. **데이터 범주화** — 성분을 기능·위험도 기준으로 구조화
2. **타겟 정의** — Good(고신뢰 성분) vs Others로 이진 분류
3. **결측치 처리** — Missing은 제외 (WoE = 0)

**사이트 신뢰도 가중치(Q값):**
관련성·타당성·신뢰성 3개 요인, 9개 문항(5점 척도)으로 6명이 평가한 설문 결과를 정규화하여 산출했습니다.

| 사이트            | 총점  | Q값            |
|----------------|-----|---------------|
| coos.kr        | 226 | 0.3419        |
| 화해             | 217 | 0.4989 (재정규화) |
| Paula's Choice | 218 | 0.5011 (재정규화) |

**Final Score 통합 공식:**


<p align="center">
 <img src="https://latex.codecogs.com/svg.image?Final%5C%20Score%20%3D%20%5Cunderbrace%7BV_%7Bcoos%7D%20%5Ctimes%20Q_%7Bcoos%7D%7D_%7BRegulatory%7D%20%2B%20%5Cunderbrace%7BWoE_%7Bhwahae%7D%20%5Ctimes%20Q_%7Bhwahae%7D%7D_%7BUser%5C%20Experience%7D%20%2B%20%5Cunderbrace%7BWoE_%7Bpaula%7D%20%5Ctimes%20Q_%7Bpaula%7D%7D_%7BExpert%7D" alt="Final Score formula"/>
</p>


최종 점수가 높을수록 안전하고 신뢰도 높은 성분이며, 낮을수록 주의·위험 성분으로 필터링됩니다. 이 구조는 임의 설계가 아닌 **설명 가능한 AI(XAI)** 원칙에 기반하여, 추천 이유를 사용자에게 자연어로 제공할
수 있습니다.


</details>


<details>
<summary><b>🔀 검색 모델 라우팅 & 실험 결과</b></summary>


LangGraph classify_node가 질문을 자동 분류한 뒤, NDCG@3 평가 기반으로 최적 검색 방식을 적용합니다.

| 질문 유형      | 선택 방식 | NDCG@3 | 예시                 |
|:-----------|:-----:|:------:|:-------------------|
| ingredient | Dense | 1.444  | "나이아신아마이드 EWG 등급?" |
| recommend  | BM25  | 1.822  | "지성 피부에 뭐 써?"      |
| general    | BM25  | 0.926  | "화장품 어떻게 보관?"      |

**FAISS 프리셋 (청크 가중치)**

|   프리셋   | EWG  | Basic Info | Expert | 용도                  |
|:-------:|:----:|:----------:|:------:|:--------------------|
| Preset1 | 0.33 |    0.33    |  0.33  | general (균등)        |
| Preset2 | 0.50 |    0.35    |  0.15  | ingredient (안전성 중심) |
| Preset3 | 0.40 |    0.45    |  0.15  | 안전성 + 기본정보 균형       |
| Preset4 | 0.45 |    0.45    |  0.10  | recommend           |

</details>


<details>
<summary><b>📁 폴더 구조</b></summary>

```
flow/
├── 00_data/
│   ├── 00_raw/          # 원본 데이터
│   ├── 01_interim/      # 전처리 중간 산출물
│   └── 02_processed/    # 최종 RAG용 청크
├── 01_notebooks/        # 실험 로그
├── 02_src/
│   ├── 00_common/       # 설정 로더, 로거
│   ├── 01_data/         # 데이터 파이프라인
│   ├── 02_model/        # FAISS 인덱싱, RAG 체인, 평가
│   └── 03_front/        # Streamlit UI
├── 03_scripts/          # 파이프라인 자동화
├── 04_configs/          # config.yaml
├── 05_artifacts/        # FAISS 인덱스
├── 06_assets/           # README 이미지
├── api_server.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

</details>


---

## 👥 Team Flow

**프로젝트 기간**: 2026.04.24 — 2026.04.27


<table>
<tr align="center">
<td><img src="https://i.namu.wiki/i/643jgTgLVQz0BSwMlgrrBHwReXJ19iRGS5bxDMLSxsPM4GkN-uOFsy6Pp9RiX7nEasn9WvHMLs09raOXZdp55UnSOTBlyHuBGlsFQUfmEKMihNHSeDJonYr23W2RjRrrLDY0wYrSCo3vvxYkZSFmVw.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/UNFQIgJYne_H9jkN5j24jyGY2laGmWrke_x3M-nEZkSD1J5wTNIRS7Wx_iJyGCYqcFMJ1aNHSn5HNlKF_8lM9_wR-zKUCdLHdDjRJ1Yn8X6nHJ9cOdwQP_obJfqsIVuIT4i90lSi1RpObI9txk28NQ.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/9ox-ZTFvJTp9NnfR7lfYejD5hQuBsARibzQva-1eZYOYFig3m4OrVnxdZXNhRdmOvHzjtC5jGb9P_IXejenqrWx6j6-kLwItI1oJE08p09mdCV2DPmhoTPPs4sOh1sdzg_GpB-koxeq_upE93UXs8w.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/4fCAU9Ybh2SEjBXvmxfujyqF8O1L7ErL8_wambUdwOsL9wOkoN_iQ9baUv1JV0xBRn33dxLfENat13pAQfDFgJ92IRe8ydxy91_YB9PRr_xehlgXqDZZZ8dtDpsoG69LeNDwvPLzvKYn8gywIlMxpQ.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/ZnpEAAI7med8T9czv4jHEO3F_SRO0vb-vuNnvRvONC898ryjJrEiG5vUAF_nuTApH9Fe2CDjEOEHq-kSIA1AvpStjcxh0h91B1iDVP3hM3QfeR7hj7K97FxKGJDiJpfGG6t6wSK5F4fbjbFPoKr8uA.webp" width="100"></td>
<td><img src="https://i.namu.wiki/i/4LeawTUtEIuFpBNrGYYmZDUfLflQiuQlvlU-sDR-BXgLVntn2krnY6XuBYPUgkOCEUqrdpoEHJqW2msV3JYWBTOAHoCFAYCAi7WW0tzSdO9uTbQJI2jLeUam-4O82pvIQ5Dnla5OvIqxb-njgjO2Uw.webp" width="100"></td>
</tr>
<tr align="center">
<td><b>김민하</b></td>
<td><b>배재현</b></td>
<td><b>윤지혜</b></td>
<td><b>전윤하</b></td>
<td><b>정다솔</b></td>
<td><b>홍진서</b></td>
</tr>
<tr align="center">
<td><a href="https://github.com/leedhroxx"><img src="https://img.shields.io/badge/GitHub-leedhroxx-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/rshyun24"><img src="https://img.shields.io/badge/GitHub-rshyun24-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/jjhhyy0926"><img src="https://img.shields.io/badge/GitHub-jjhhyy0926-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/yoonha315"><img src="https://img.shields.io/badge/GitHub-yoonha315-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/soll07"><img src="https://img.shields.io/badge/GitHub-soll07-181717?style=flat-square&logo=github" /></a></td>
<td><a href="https://github.com/Hong-Jin-seo"><img src="https://img.shields.io/badge/GitHub-Hong--Jin--seo-181717?style=flat-square&logo=github" /></a></td>
</tr>
<tr align="center">
<td>데이터 전처리<br>4가지 검색 병렬 실행<br>이중 가중치 재정렬<br>GitHub 총괄</td>
<td>Chunk 변환 & 임베딩<br>PPT 작성<br>기획 및 발표</td>
<td>데이터 수집<br>Langchain & RAGChain<br>Streamlit UI</td>
<td>Chunk 변환 & 임베딩<br>README 작성</td>
<td>데이터 수집<br>Langchain & RAGChain<br>Streamlit UI</td>
<td>데이터 전처리<br>4가지 검색 병렬 실행<br>이중 가중치 재정렬<br>Notion 총괄</td>
</tr>
</table>


---

## 💬 회고

> 여기에 팀원별 또는 전체 회고를 작성해주세요.


---

## 📚 데이터 출처 & 라이선스

| 소스                                                                   | 제공 정보                    |
|:---------------------------------------------------------------------|:-------------------------|
| [coos.kr](https://coos.kr)                                           | 성분별 기능, 국가별 규제, AI 설명    |
| [화해](https://www.hwahae.co.kr)                                       | EWG 수치, 사용자 리뷰 토픽, 피부 타입 |
| [Paula's Choice](https://www.paulaschoice.com/ingredient-dictionary) | 전문가 평가 및 논문 근거           |

본 프로젝트는 [MIT License](./LICENSE) 하에 배포됩니다.

