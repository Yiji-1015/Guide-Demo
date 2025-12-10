# backend/app.py
import os
# from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
import streamlit as st

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client_oa = OpenAI(api_key=OPENAI_API_KEY)
# Qdrant 클라이언트 (로컬)
client_qd = QdrantClient(
    url="http://localhost:6333",  # 또는 host="localhost", port=6333
)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
COLLECTION_NAME = "lloydk_docs"


def embed_batch(texts):
    """
    texts: list[str]
    return: list[list[float]] (임베딩 벡터들)
    """
    cleaned = []
    idx_map = []

    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if not t:
            continue
        cleaned.append(t)
        idx_map.append(i)

    if not cleaned:
        return [], []

    resp = client_oa.embeddings.create(
        model=EMBED_MODEL,
        input=cleaned,
    )

    vectors = [d.embedding for d in resp.data]
    return vectors, cleaned


def rag_answer(question: str) -> str:
    """
    한 번 질문 -> 한 번 답변 반환 (스트림릿/웹에서 사용하기 좋게)
    """
    question = question.strip()
    if not question:
        return "질문이 비어 있어요. 뭘 물어볼지 입력해 주세요 🙂"

    # 1️⃣ 쿼리 임베딩
    q_emb, _ = embed_batch([question])
    if not q_emb:
        return "임베딩을 생성하지 못했어요. 질문을 다시 한번 입력해 주세요."

    # 2️⃣ Qdrant에서 검색
    results = client_qd.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb[0],
        limit=5,
        with_payload=True
    )

    # 3️⃣ 검색된 문서 모으기
    contexts = [r.payload["text"] for r in results.points if "text" in r.payload]
    if not contexts:
        return "관련 문서를 찾지 못했어요. 질문을 조금 다르게 해볼까요? 🤔"

    # 4️⃣ 답변 생성
    context_text = "\n\n".join(contexts)
    system_prompt = (
        "너는 DO 솔루션 관련 기술 문서를 바탕으로 답변하는 어시스턴트야.\n"
        "반드시 아래 제공된 문맥 내에서만 답변해. 모르면 모른다고 말해.\n"
        "table의 경우, [[열1, 열2], [열1에 대한 아이템, 열2에 대한 아이템]...] 이런 식으로 되어있으니 반드시 끝까지 다 보고 답변해야 해."
    )
    user_prompt = (
        f"[질문]\n{question}\n\n[관련 문서]\n{context_text}\n\n"
        "위 내용을 기반으로 한국어로 자연스럽게 답변해줘."
    )

    resp = client_oa.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    return answer


def rag_chat():
    """
    터미널에서 테스트용 CLI 버전 (원래 있던 거 유지하고 싶으면)
    """
    print("RAG 챗봇입니다. 'exit' 입력 시 종료.\n")
    while True:
        q = input("👤 질문: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            print("bye~")
            break

        answer = rag_answer(q)
        print("\n🤖 답변:\n", answer, "\n")


if __name__ == "__main__":
    # 터미널에서 실행하면 CLI 모드로
    rag_chat()