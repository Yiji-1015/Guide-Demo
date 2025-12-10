# frontend/app.py
import streamlit as st
import sys
import os

# 백엔드 모듈 import 경로 설정 (프로젝트 구조에 따라 조정)
# 예: 프로젝트 루트/
#     ├─ backend/
#     │    └─ app.py (여기에 rag_answer 있음)
#     └─ frontend/
#          └─ app.py (이 파일)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.app import rag_answer  # ✅ 방금 만든 함수 가져오기


st.set_page_config(page_title="DO 솔루션 RAG 챗봇")

st.title("DO 솔루션 RAG 챗봇")
st.write("DO 관련 기술 문서 기반으로 답변해주는 챗봇입니다 🙂")

# 세션에 대화 히스토리 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 표시
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# 사용자 입력
user_input = st.chat_input("무엇이 궁금하신가요?")

if user_input:
    # 1) 화면에 유저 메시지 추가
    st.session_state["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 모델 호출
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer = rag_answer(user_input)
            st.markdown(answer)

    # 3) 히스토리에 어시스턴트 답변 저장
    st.session_state["messages"].append(("assistant", answer))

# streamlit run app.py 