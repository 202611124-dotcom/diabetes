from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename):
    artifact_path = BASE_DIR / filename
    if not artifact_path.exists():
        message = (
            f"필수 파일을 찾을 수 없습니다: {filename}\n"
            f"예상 위치: {artifact_path}\n"
            "프로젝트 루트에 모델 파일을 넣어주세요."
        )
        st.error(message)
        return None
    return joblib.load(str(artifact_path))

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="당뇨 예측 시스템",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# CSS 디자인
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f9ff;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1565c0;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #1976d2, #42a5f5);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1565c0, #1e88e5);
}

.result-box {
    background-color: #e3f2fd;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
}

.result-text {
    font-size: 38px;
    font-weight: bold;
    color: #0d47a1;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 모델 및 스케일러 로드
# -----------------------------
knn_model_eng = load_artifact("knn_model_eng.pkl")
scaler = load_artifact("scaler.pkl")
X_columns = load_artifact("X_columns.pkl")
X_mean = load_artifact("X_mean.pkl")

# -----------------------------
# 제목
# -----------------------------
st.markdown('<div class="title">🩺 AI 당뇨 예측 시스템</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">건강 데이터를 기반으로 당뇨 여부를 예측합니다.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# 입력 영역
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📋 건강 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    total_chol = st.number_input("총콜레스테롤", value=200.0)
    stab_glu = st.number_input("공복혈당", value=90.0)
    hdl = st.number_input("HDL콜레스테롤", value=40.0)
    ratio = st.number_input("콜레스테롤비율", value=5.0)
    glyhb = st.number_input("당화혈색소", value=5.5)

with col2:
    age = st.number_input("나이", value=45)
    gender = st.selectbox("성별", ["female", "male"])
    height = st.number_input("키(cm)", value=165.0)
    weight = st.number_input("몸무게(kg)", value=70.0)
    waist = st.number_input("허리둘레", value=35.0)

with col3:
    hip = st.number_input("엉덩이둘레", value=40.0)
    bp_1s = st.number_input("1차 수축기혈압", value=130.0)
    bp_1d = st.number_input("1차 이완기혈압", value=80.0)
    bp_2s = st.number_input("2차 수축기혈압", value=140.0)
    bp_2d = st.number_input("2차 이완기혈압", value=90.0)
    time_ppn = st.number_input("식후경과시간(분)", value=180.0)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 예측 버튼
# -----------------------------
if st.button("🧠 당뇨 예측하기"):

    # -----------------------------
    # 입력 데이터 생성
    # -----------------------------
    input_data_raw = pd.DataFrame([
        [
            total_chol,
            stab_glu,
            hdl,
            ratio,
            glyhb,
            age,
            gender,
            height,
            weight,
            bp_1s,
            bp_1d,
            bp_2s,
            bp_2d,
            waist,
            hip,
            time_ppn
        ]
    ],
        columns=[
            '총콜레스테롤',
            '공복혈당',
            'HDL콜레스테롤',
            '콜레스테롤비율',
            '당화혈색소',
            '나이',
            '성별',
            '키',
            '몸무게',
            '1차_수축기혈압',
            '1차_이완기혈압',
            '2차_수축기혈압',
            '2차_이완기혈압',
            '허리둘레',
            '엉덩이둘레',
            '식후경과시간'
        ]
    )

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    input_data_raw['복부비만도'] = (
        input_data_raw['허리둘레'] /
        input_data_raw['엉덩이둘레']
    )

    input_data_raw['BMI'] = (
        input_data_raw['몸무게'] /
        ((input_data_raw['키'] / 100) ** 2)
    )

    input_data_raw['나쁜콜레스테롤_비율'] = (
        (input_data_raw['총콜레스테롤'] -
         input_data_raw['HDL콜레스테롤']) /
        input_data_raw['HDL콜레스테롤']
    )

    # -----------------------------
    # 결측치 처리
    # -----------------------------
    for col in input_data_raw.columns:
        if pd.api.types.is_numeric_dtype(input_data_raw[col]):
            if input_data_raw[col].isnull().any():
                input_data_raw[col] = input_data_raw[col].fillna(
                    X_mean.get(col, 0)
                )

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    input_data_encoded = pd.get_dummies(
        input_data_raw,
        columns=['성별'],
        drop_first=True
    )

    # -----------------------------
    # 컬럼 맞추기
    # -----------------------------
    final_input_data = pd.DataFrame(
        columns=X_columns,
        index=[0]
    ).fillna(0)

    for col in input_data_encoded.columns:
        if col in final_input_data.columns:
            final_input_data[col] = input_data_encoded[col]

    # -----------------------------
    # 스케일링
    # -----------------------------
    input_scaled = scaler.transform(final_input_data)

    # -----------------------------
    # 예측
    # -----------------------------
    predicted = knn_model_eng.predict(input_scaled)
    prob = knn_model_eng.predict_proba(input_scaled)

    diabetes_prob = prob[0][1] * 100

    # -----------------------------
    # 결과 출력
    # -----------------------------
    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:24px;">당뇨 예측 결과</div>
        <div class="result-text">
            {"당뇨 위험" if predicted[0] == 1 else "정상"}
        </div>
        <div style="font-size:26px; margin-top:10px;">
            위험 확률 : {diabetes_prob:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # 상태 메시지
    # -----------------------------
    if diabetes_prob >= 70:
        st.error("⚠ 높은 당뇨 위험군입니다.")
    elif diabetes_prob >= 40:
        st.warning("⚠ 주의가 필요한 상태입니다.")
    else:
        st.success("✅ 비교적 정상 범위입니다.")