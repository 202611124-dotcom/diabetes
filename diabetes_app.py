import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 모델과 스케일러 불러오기용

# --- 0. 모델 및 데이터 준비 (사전 학습된 객체가 있다고 가정) ---
# 실제 환경에서는 학습된 모델, 스케일러, X의 컬럼명 리스트 등을 파일로 저장해두고 불러와야 합니다.
# 예: knn_model = joblib.load('knn_model.pkl')
# 여기서는 코드 구조에 집중하여 작성합니다.

st.set_page_config(page_title="당뇨병 예측 시스템", layout="wide")

st.title("🩺 당뇨병 발병 예측 시스템")
st.markdown("사용자의 건강 지표를 입력하여 당뇨 가능성을 확인하세요.")
st.divider()

# --- 1. 사용자 입력 받기 (Layout 구성) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("기본 정보")
    age = st.number_input("나이 (예: 45)", min_value=0, max_value=120, value=45)
    gender = st.selectbox("성별", ["female", "male"])
    height = st.number_input("키(인치)", value=65.0)
    weight = st.number_input("몸무게(파운드)", value=180.0)

with col2:
    st.subheader("혈액 지표")
    total_chol = st.number_input("총콜레스테롤", value=200.0)
    stab_glu = st.number_input("공복혈당", value=90.0)
    hdl = st.number_input("HDL콜레스테롤", value=40.0)
    ratio = st.number_input("콜레스테롤비율", value=5.0)
    glyhb = st.number_input("당화혈색소", value=5.5)

with col3:
    st.subheader("혈압 및 체격")
    bp_1s = st.number_input("1차 수축기혈압", value=130.0)
    bp_1d = st.number_input("1차 이완기혈압", value=80.0)
    
    # 2차 혈압 (결측치 처리를 위해 체크박스 활용 가능)
    use_bp2 = st.checkbox("2차 혈압 측정치 있음")
    if use_bp2:
        bp_2s = st.number_input("2차 수축기혈압", value=130.0)
        bp_2d = st.number_input("2차 이완기혈압", value=80.0)
    else:
        bp_2s = np.nan
        bp_2d = np.nan
        
    waist = st.number_input("허리둘레", value=35.0)
    hip = st.number_input("엉덩이둘레", value=40.0)
    time_ppn = st.number_input("식후경과시간 (분)", value=180.0)

# --- 2. 데이터 전처리 ---
# 버튼 클릭 시 예측 실행
if st.button("예측하기", type="primary"):
    
    # 입력 데이터를 DataFrame으로 변환
    input_dict = {
        '총콜레스테롤': [total_chol], '공복혈당': [stab_glu], 'HDL콜레스테롤': [hdl],
        '콜레스테롤비율': [ratio], '당화혈색소': [glyhb], '나이': [age], '성별': [gender],
        '키': [height], '몸무게': [weight], '1차_수축기혈압': [bp_1s], '1차_이완기혈압': [bp_1d],
        '2차_수축기혈압': [bp_2s], '2차_이완기혈압': [bp_2d], '허리둘레': [waist],
        '엉덩이둘레': [hip], '식후경과시간': [time_ppn]
    }
    input_data_raw = pd.DataFrame(input_dict)

    # 파생 변수 생성
    input_data_raw['복부비만도'] = input_data_raw['허리둘레'] / input_data_raw['엉덩이둘레']
    input_data_raw['BMI'] = input_data_raw['몸무게'] / (input_data_raw['키'] / 100)**2
    input_data_raw['나쁜콜레스테롤_비율'] = (input_data_raw['총콜레스테롤'] - input_data_raw['HDL콜레스테롤']) / input_data_raw['HDL콜레스테롤']

    # 결측치 처리 (학습 데이터의 평균값을 사용해야 함 - 예시로 0 처리)
    # 실제 운영시에는 미리 계산된 평균값 딕셔너리를 사용하세요.
    input_data_raw = input_data_raw.fillna(0) 

    # One-Hot Encoding
    input_data_encoded = pd.get_dummies(input_data_raw, columns=['성별'])
    # '성별_male'만 남기고 drop_first=True 효과 내기 (X_columns에 맞춰 조정 필요)
    if '성별_female' in input_data_encoded.columns:
        input_data_encoded = input_data_encoded.drop(columns=['성별_female'])

    # --- 컬럼 순서 맞추기 ---
    # 실제 환경에서는 X.columns.tolist()를 저장해둔 리스트를 사용해야 합니다.
    # 여기서는 변수 X가 세션에 있다고 가정하거나, 학습 시의 순서를 리스트로 정의하세요.
    try:
        # 학습된 모델의 피처 순서에 맞게 재배열
        # final_input_data = input_data_encoded.reindex(columns=X_columns, fill_value=0)
        
        # --- 3. 스케일링 및 예측 ---
        # input_scaled = scaler.transform(input_data_encoded)
        # predicted = knn_model_eng.predict(input_scaled)
        # prob = knn_model_eng.predict_proba(input_scaled)

        # 결과 출력 (더미 결과로 대체 - 실제 모델 연결 시 수정)
        st.divider()
        st.subheader("📊 예측 결과")
        
        # 예시 출력을 위한 임의값 (실제 코드에서는 predicted[0], prob[0][1] 사용)
        res_val = 0  # 임시
        res_prob = 0.15 # 임시
        
        if res_val == 0:
            st.success(f"결과: **정상** (당뇨 확률: {res_prob*100:.1f}%)")
        else:
            st.error(f"결과: **당뇨 위험** (당뇨 확률: {res_prob*100:.1f}%)")
            
    except NameError:
        st.warning("모델(`knn_model_eng`)이나 스케일러(`scaler`)가 로드되지 않았습니다. 학습된 객체를 연결해주세요.")