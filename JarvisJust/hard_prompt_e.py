# 처음 보여지는 문구
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

# 항상 state의 맨 앞에 넣는 문구
def generate_state_message(user_name):
    state_message = {"role": "system",
                 "content": f"""You are a matchmaker, And the user's name is {user_name}.
                 Remember this, Refer to the chatting history below, and Answer the user's message in Korean."""}
    return state_message

# 의도 파악 prompt
def classify_prompt(msg):
    prompt = f"""As a professional matchmaker, you specialize in matching the user to a person, services, products, businesses,
                    and anything to meet his(her) needs and preferences.
                    Here is the user's message: {msg}
                    The message can be classified as 2 types: "Match", "Chat"
                    Rephrase the message in standard language, think about what the user needs or wants, and answer what type the message is.

                    * Tips for classification:
                    1. "Match" has really broad meaning and includes social interaction, connection, recommendation, etc.
                    2. If the message has 'object', it is likely to be "Match".
                    3. If the message wants to be matched or connected someone or something, it is likely to "Match".
                    4. If the message finds a place to buy, pack, or sell something, it is likely to "Match".
                    5. If the object is about another person or another thing(such as '다른 사람'), it must be "Chat".
                    6. If the message is about the user's feeling, thought, or opinion, it is likely to be "Chat".

                    * Cautions:
                    1. If you cannot understand or classify the message, it is "Chat".
                    2. Only answer like this: ["type", "one object"]
                    (If the type is "Chat", Must not answer the object.)
                    3. The language of the object should follow the message."""
    return prompt

# 채팅 프롬프트
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

# RAG 프롬프트
RAG_prompt = """
            Your task is to perform the following actions:
            다음 동작을 수행하세요:
            1 - <>로 구분된 다음 텍스트에서 사용자의 요구사항을 정확히 파악합니다.
            2 - 매칭리스트로부터 사용자의 요구사항을 충족시킬 수 있는 다른 사용자와 매칭합니다.
                매칭리스트는 (사용자, 메시지) 형식으로 구성되어 있으며, 사용자명이 매칭에 영향을 주어서는 안 됩니다.
            3 - 매칭된 사용자가 요구사항을 충족시킬 수 있는 이유를 제공합니다.
            4 - 사용자의 질문에 알맞게 답변의 맥락을 조정해서 답변하세요.

            다음 형식을 사용하십시오:
            사용자의 요구사항을 바탕으로 다음 에이전트를 추천합니다.

            Agent: Agent
            메시지: 메시지
            매칭이유: 매칭이유

            각 답변을 줄바꿈 문자로 구분하세요.
            답변에는 maximum 2명의 Agent만을 제시합니다.
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
                    2 - 조금이라도 연관이 있을 경우 매칭해야 합니다.
                    3 - 답변에는 1명의 Agent만을 제시합니다.
                    4 - 중복된 사용자를 매칭하면 안 됩니다.
                    5 - 답변 형식은 "Agent명" 입니다.
                    6 - 만약 매칭리스트에 적절한 Agent가 포함되어 있지 않다면, 답변 형식은 "None" 입니다.

                    매칭리스트: {profile}"""
    return RAG_prompt

def one_agent_rag_prompt(one_agent, one_message):
    one_RAG_prompt = f"""
                    Quickly match an agent from a provided list with specific criteria. Here's what you need to do:

                    1. {one_agent}: Agent
                    2. {one_message}: 메시지
                    3. Answer the <매칭 이유> based on the information above. 

                    Format your response as follows:
                    사용자의 요구사항을 바탕으로 다음 에이전트를 추천합니다.

                    Agent: [Recommended Agent]
                    메시지: [Agent's Message]
                    매칭 이유: [Reason for Match]

                    *Cautions:
                    - Only include one recommended agent.
                    - Must answer the reason for matching as concisely as possible in a single sentence. (100 characters less)
                    """
    return one_RAG_prompt