
def generate_welcome_message(user_name):
    opening_welcome = f"""안녕하세요? {user_name}님. 
                        저는 JarvisJust(JJ)입니다. 무엇을 도와드릴까요? 

                        찾으시는 내용을 아래처럼 입력해주시면 JJ가 연결해드립니다.
                        예1) 진미채 살 수 있는 곳을 찾아줘
                        예2) 주말에 축구 같이할 사람 찾아줘

                        연결된 이후 상품 구매를 희망하시면,                                        
                        ‘닉네임과 상품 결제해줘’를
                        예) OOO(닉네임) 상품 결제해줘

                        1:1채팅 연결을 원하시면, 
                        ‘닉네임과 연결해줘’를 입력해주세요.
                        예) OOO(닉네임) 연결해줘"""
    return opening_welcome

def generate_state_message(user_name):
    state_message = {"role": "system",
                 "content": f"""You are a matchmaker, And the user's name is {user_name}.
                 Remember this, Refer to the chatting history below, and Answer the user's message in Korean."""}
    return state_message

## 3/22 수정(윤이지)
def classify_prompt(msg):
    prompt = f"""As a professional matchmaker, you specialize in matching the user to a person, services, products, businesses,
                    and anything to meet his(her) needs and preferences.
                    Here is the user's message: {msg}
                    The message can be classified as 3 types: "Chat", "Buy&Sell", "Match"
                    Rephrase the message in standard language, think about what the user needs or wants, and answer what type the message is.

                    * Tips for classification:
                    1. "Match" has really broad meaning and includes social interaction, connection, recommendation, etc.
                    2. If the message has 'object', it is likely to "Buy&Sell" or "Match".
                    3. If the message wants to be matched or connected someone or something, it is likely to "Match".
                    4. If the message finds a place to buy, pack, or sell something, it is likely to "Buy&Sell".
                    5. If the object is 'other person' or 'other thing', it is likely to "Chat".
                    6. If the message includes "다른", it is likely to "Chat".

                    * Cautions:
                    1. If you cannot understand or classify the message, it is "Chat".
                    2. Only answer like this: ["type", "one object"]
                    (If the type is "Chat", Must not answer the object.)
                    3. The language of the object should follow the message."""
    return prompt

## 3/22 수정(윤이지)
purchase_prompt = [
    {"question": "세차 서비스로 차량을 새 것처럼 만들어 드리겠습니다. 세차에 관한 선호 사항을 알려주세요.",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: This message is not about selling products.
    So the final answer should be like this form: ['no-product']"""},

    {"question": "노트북 사고 싶어.",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: This message is not about selling products.
    So the final answer should be like this form: ['no-product']"""},

    {"question": "백진미와 홍진미 오징어채 팔아요",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: 백진미와 홍진미 오징어채
    What is the price: There is no price in the message, so the answer is 0
    What is the quantity: There is no quantity in the message, so the answer is ''
    So the final answer should be like this form: ['백진미와 홍진미 오징어채', 0, '']"""},

    {"question": "맛있고 고단백의 간식을 찾고 계신가요? 코주부의 육포 大 (900g) 제품이 여러분의 기대를 충족시켜드릴 건강한 간식입니다! 이제 중부시장(서울특별시 중구 을지로30길 29)에서 12000원에 바로 만나보세요!",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: 코주부의 육포 大
    What is the price: 12000원
    What is the quantity: 900g
    So the final answer should be like this form: ['코주부의 육포 大', 12000, '900g']"""},

    {"question": "쫄깃한 식감이 매력적인 더욱 도톰해진 쥐치포 10미를 경험해보세요! 서울의 맛을 그대로 담아낸 이 특별한 간식을 찾는다면, 중부시장(서울특별시 중구 을지로30길 29)에서 만나볼 수 있습니다.",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: 쥐치포
    What is the price: There is no price in the message, so the answer is 0
    What is the quantity: 10미
    So the final answer should be like this form: ['쥐치포', 0, '10미']"""},

    {"question": "맥반석 오징어 구이는 술안주나 군것질에 이상적이며 아빠와 아이들 간식으로도 강력 추천합니다! 25000원으로, 맛있고 건강한 간식을 찾으시는 분들은 중부시장(서울특별시 중구 을지로30길 29)에서 만나보세요.",
    "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: 맥반석 오징어 구이
    What is the price: 25000원
    What is the quantity: There is no quantity in the message, so the answer is '
    So the final answer should be like this form: ['맥반석 오징어 구이', 25000, '']"""},

    {"question": "홍진미 400g 국내 가공 제품으로 순수한 맛과 품질을 경험하세요! 고객님의 건강을 생각한 최상의 제품을 직접 만나보실 수 있습니다. 지금 바로 중부시장(서울특별시 중구 을지로30길 29)에서 만나보세요.",
     "answer": """
    Are follow up questions needed here: Yes.
    What is the product name: 홍진미, You should answer only the product name, not the quantity.
    What is the price: There is no price in the message, so the answer is 0
    What is the quantity: 400g
    So the final answer should be like this form: ['홍진미', 0, '400g']"""}
    ]

## 3/22 수정(윤이지)
def get_chat_prompt(msg):
    chat_prompt =f"""
            Your task is to perform the following actions:
            다음 동작을 수행하세요:
            1 - 매칭 리스트를 기반으로하여, 사용자의 질문에 대해 최대한 상세하게 답변합니다.
            2 - 답변 시 가능하다면 'role':'assistant'의 마지막 'content'를 reference하되, 
            그것을 reference했다는 사실을 답변에 절대 포함하지 않습니다.
            3 - <>로 구분된 다음 텍스트가 영어라면 모든 답변을 영어로 번역하여 제공합니다.

            사용자의 메시지:<{msg}>,

            * Cautious:
            1 - 없는 사실을 지어내지 마십시오.
            2 - 최종 답변에서 다른 사람의 닉네임이 있는 경우, 'Agent: 닉네임\n메시지: Agent의 메시지' 형식으로 출력하라"""
    return chat_prompt

RAG_prompt = """
            Your task is to perform the following actions:
            다음 동작을 수행하세요:
            1 - <>로 구분된 다음 텍스트에서 사용자의 요구사항을 정확히 파악합니다.
            2 - 매칭리스트로부터 사용자의 요구사항을 충족시킬 수 있는 다른 사용자와 매칭합니다.
                매칭리스트는 (사용자, 메시지) 형식으로 구성되어 있으며, 사용자명이 매칭에 영향을 주어서는 안 됩니다.
            3 - 매칭된 사용자가 요구사항을 충족시킬 수 있는 이유를 제공합니다.

            다음 형식을 사용하십시오:
            사용자의 요구사항을 바탕으로 다음 에이전트를 추천합니다.

            Agent: Agent
            메시지: 메시지
            매칭이유: 매칭이유

            각 답변을 줄바꿈 문자로 구분하세요.
            답변에는 1~2명의 Agent만을 제시합니다.
            매칭 이유는 최대한 간결하게 답변하세요.
            조금이라도 연관이 있을 경우 매칭해야 합니다.
            중복된 사용자를 매칭하면 안 됩니다.

            만약 매칭리스트에 적절한 사용자가 포함되어 있지 않다면,
            간단히 "죄송합니다, 현재 요청하신 매칭 조건에 적합한 사용자를 찾지 못하였습니다. 원하시는 다른 매칭이 있으신가요?"라고 작성하세요.
            """


def matching_rag_prompt(msg, profile):
    RAG_prompt = f"""
                    * Your task is to perform the following actions:
                    1 - <>로 구분된 다음 텍스트에서 사용자의 요구사항을 정확히 파악합니다.
                    2 - 사용자의 요구사항을 충족시킬 수 있는 다른 Agent와 매칭합니다. 예를 들어, 연필을 사고 싶어 하는 사용자에게 연필을 판매하는 Agent와 매칭해야 합니다.
                    3 - 매칭된 Agent가 요구사항을 충족시킬 수 있는 이유를 생각해 보세요.
                    <{msg}>

                    * Cautious:
                    1 - 매칭리스트는 (Agent, 메시지) 의 형태로 되어있습니다. Agent명은 매칭 시 고려해서는 안 됩니다.
                    2 - 매칭 이유는 최대한 간결하게 답변하세요.
                    3 - 조금이라도 연관이 있을 경우 매칭해야 합니다.
                    4 - 답변에는 1명의 Agent만을 제시합니다.
                    5 - 중복된 사용자를 매칭하면 안 됩니다.
                    6 - 답변 형식은 "Agent명" 입니다.
                    7 - 만약 매칭리스트에 적절한 Agent가 포함되어 있지 않다면, 답변 형식은 "None" 입니다.

                    매칭리스트: {profile}"""
    return RAG_prompt