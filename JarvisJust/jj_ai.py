import ast
import re
import json
from openai import BadRequestError 
from kiwipiepy.utils import Stopwords # 한국어 자연어 처리 패키지
from kiwipiepy import Kiwi
import faiss
import numpy as np

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI as OpenAI2
from decouple import config
from .hard_prompt import classify_prompt, purchase_prompt, get_chat_prompt, RAG_prompt, matching_rag_prompt
from .db_utils import all_data_from_db, keyword_search, vectors_list, product_ids_list, extract_sentences
import asyncio




# m1 s1 ss1
# 문장으로부터 상품명, 가격, 수량 추출 (가격 default=38000원)
async def purchase_keyword(msg):
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="question: {question}\nanswer: {answer}"
    )
    prompt = FewShotPromptTemplate(
        examples=purchase_prompt,
        example_prompt=example_prompt,
        suffix="question : {question}",
        input_variables=["question"],
    )
    llm = OpenAI2(openai_api_key=config('OPENAI_API_KEY'))
    parser = StrOutputParser()
    chain = prompt | llm | parser
    chain_answer = chain.invoke({"question": msg})
    answer_list = ast.literal_eval(str(chain_answer[chain_answer.find('form: ')+6:]))

    if len(answer_list) == 1:
        return 'no_product'
    elif len(answer_list) == 2:
        if answer_list[0] == 'no_product' or answer_list[0] == '':
            return 'no_product'
        else:
            return answer_list
    elif answer_list[1] == '' and answer_list[2] == '':
        return 'no_product'
    else:
        return 'no_product'


# m1 s1
## 3/28 수정(2)
async def classify_process(client, text):
    prompt = f"""If the sentence({text}) seems to pay or something with certainty, only print 'Buy&Sell'.
                If the sentence seems to be connected or matched someone, only print 'Match'.
                If you cannot judge, or the sentence is a kind of questions or seemes to wnat info, only print out 'Chat'."""
    response = client.chat.completions.create(
                    model = "gpt-4-1106-preview",
                    messages=[{"role": "system", "content": prompt}],
                    temperature = 0
                )
    response = response.choices[0].message.content
    return response


## 3/22 추가(윤이지) - RAG에 json 형태로 변환
async def RAG_to_json(user_name, msg, client):
    model="text-embedding-3-large"
    search_vector = client.embeddings.create(input=[msg], model=model).data[0].embedding
    return {
        "user_name": user_name,
        "msg": msg,
        "vector": search_vector
        }
      
## 3/22 추가(윤이지) - 와/과 구분하는 함수
def ends_with_jong(kstr):
    k = kstr[-1]
    if "가" <= k <= "힣":
        if (ord(k)-ord("가")) % 28 > 0:
            return "과"
        else:
            return "와"
    else:
        return "와"

# m1
## 3/22 수정(윤이지)
async def message_process(client, msg, state, matched_agent, matched_msg):
    state.append({"role": "user", "content": msg})

    if not msg:
        empty_msg = '대화를 입력해주세요.'
        state.append({"role": "assistant", "content": empty_msg})
        result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
        result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": empty_msg}
        return "end", result_ME, result_AI, state

    else:
        for i in range(len(matched_agent)):
            if matched_agent[i] in msg.replace(" ", ""):
                classified_process = await classify_process(client, msg)

                # 채팅창 연결 process로 이동, 최종 연결된 Agent 및 메시지 return
                if classified_process == 'Match':
                    consonant = ends_with_jong(matched_agent[i])
                    connection_msg = f"네 {matched_agent[i]+consonant} 연결해드리겠습니다."
                    state.append({"role": "assistant", "content": connection_msg})
                    message = matched_msg[i]
                    connection_keyword = await purchase_keyword(message)
                    if connection_keyword == 'no_product':
                        result_ME = { "ai_flag": "M", "type": 'connect', "mem_nick_name": matched_agent[i], "product_name": '', "product_price": 0, "talk_msg": msg}
                        result_AI = { "ai_flag": "A", "type": 'connect', "mem_nick_name": matched_agent[i], "product_name": '', "product_price": 0, "talk_msg": connection_msg}
                    else:
                        result_ME = { "ai_flag": "M", "type": 'connect', "mem_nick_name": matched_agent[i], "product_name": connection_keyword[0], "product_price": connection_keyword[1], "talk_msg": msg}
                        result_AI = { "ai_flag": "A", "type": 'connect', "mem_nick_name": matched_agent[i], "product_name": connection_keyword[0], "product_price": connection_keyword[1], "talk_msg": connection_msg}
                    return "end", result_ME, result_AI, state

                # 결제 process로 이동, 최종 연결된 Agent 및 메시지 return
                elif classified_process == 'Buy&Sell':
                    ## 3/22 수정(윤이지): 여기 db에서 꺼내오는걸로 수정 > Agent명 중복 방지 위해 matched_msg로 수정
                    message = matched_msg[i]
                    product_keyword = await purchase_keyword(message)
                    if product_keyword == 'no_product':
                        answer_msg = '결제할 상품이 없습니다.'
                        state.append({"role": "assistant", "content": answer_msg})
                        result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                        result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": answer_msg}
                        return "end", result_ME, result_AI, state
                    else:
                        pay_nickname = matched_agent[i]
                        pay_product_name =  product_keyword[0]
                        pay_product_price = product_keyword[1]
                        # pay_product_quantity = product_keyword[2]
                        pay_msg = "결제를 진행합니다"
                        state.append({"role": "assistant", "content": pay_msg})
                        result_ME = { "ai_flag": "M", "type": 'sale', "mem_nick_name": pay_nickname, "product_name": pay_product_name, "product_price": pay_product_price, "talk_msg": msg}
                        result_AI = { "ai_flag": "A", "type": 'sale', "mem_nick_name": pay_nickname, "product_name": pay_product_name, "product_price": pay_product_price, "talk_msg": pay_msg}
                    return "end", result_ME, result_AI, state
                
                ## 3/28 수정(2)
                elif classified_process == 'Chat':
                    state.append({"role": "system", "content": f"{msg}에 한국어로 답변하고, 없는 말을 지어내지 마라."})
                    response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=state,
                    stream=False)
                    response = response.choices[0].message.content
                    result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                    result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}                   
                    return "end", result_ME, result_AI, state
                
    ## 3/22 수정(윤이지) - '연결해줘'만 쳤을 때 가장 최근 Agent와 연결되는것이 잘 되지 않아 지난번과 비슷하게 수정하였습니다
    if "연결" in msg or "매칭" in msg:
        consonant = ends_with_jong(matched_agent[-1])
        connection_msg = f"네 {matched_agent[-1]+consonant} 연결해드리겠습니다."
        state.append({"role": "assistant", "content": connection_msg})
        connection_keyword = await purchase_keyword(matched_msg[-1])
        if connection_keyword == 'no_product':
            result_ME = { "ai_flag": "M", "type": 'connect', "mem_nick_name": matched_agent[-1], "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'connect', "mem_nick_name": matched_agent[-1], "product_name": '', "product_price": 0, "talk_msg": connection_msg}
        else:
            result_ME = { "ai_flag": "M", "type": 'connect', "mem_nick_name": matched_agent[-1], "product_name": connection_keyword[0], "product_price": connection_keyword[1], "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'connect', "mem_nick_name": matched_agent[-1], "product_name": connection_keyword[0], "product_price": connection_keyword[1], "talk_msg": connection_msg}
        return "end", result_ME, result_AI, state
    
    elif "결제" in msg or "구매할래" in msg:
        message = matched_msg[-1]
        product_keyword = await purchase_keyword(message)
        if product_keyword == 'no_product':
            answer_msg = '결제할 상품이 없습니다.'
            state.append({"role": "assistant", "content": answer_msg})
            result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": answer_msg}
            return "end", result_ME, result_AI, state
        
        else:
            pay_nickname = matched_agent[-1]
            pay_product_name =  product_keyword[0]
            pay_product_price = product_keyword[1]
            # pay_product_quantity = product_keyword[2]
            pay_msg = "결제를 진행합니다"
            state.append({"role": "assistant", "content": pay_msg})
            result_ME = { "ai_flag": "M", "type": 'sale', "mem_nick_name": pay_nickname, "product_name": pay_product_name, "product_price": pay_product_price, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'sale', "mem_nick_name": pay_nickname, "product_name": pay_product_name, "product_price": pay_product_price, "talk_msg": pay_msg}
        return "end", result_ME, result_AI, state
    
    # 통상적인 경우 "next" 문자열을 통해 다음 process 실행
    else:
        return "next", '-', '-', state

kiwi = Kiwi(typos="basic", model_type='sbg')
stopwords = Stopwords()
stopwords.remove(('사람', 'NNG'))

# m2 s1 ss1
async def tokenize_N(text):
    split_s = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form for i in split_s if i.tag == "NNG" or i.tag == "NNP"]
    split_list = [i.form for i in split_s]
    split_f = ','.join(split_list).replace(",", " ")
    return split_f, N_list



def faiss_index(vectors_list_json):
    vectors = np.array([json.loads(vector) for vector in vectors_list_json if vector], dtype=np.float32)
    if vectors.size == 0:
        vectors = np.zeros((1, 3072), dtype=np.float32)

    index = faiss.IndexFlatL2(vectors.shape[1]) 
    index.add(vectors)
    return index

index = faiss_index(vectors_list)

# m2 s1 ss4 sss1
async def to_vector(client, text):
    text = text.replace("\n", " ")
    model="text-embedding-3-large"
    search_vector = client.embeddings.create(input=[text], model=model).data[0].embedding
    _vector = np.array([search_vector]).astype(np.float32)
    return _vector


# m2 s1 ss4
async def search_vector(client, nickname, sentence, k1, k2, final_list):
    list_ = []
    vector = await to_vector(client, sentence)
    _, indices = index.search(vector, k=k1)

    extracted_info = await extract_sentences(indices, exclude_nickname=None)
    for info in extracted_info[:k2]:
        if info not in final_list:
            list_.append(info)
    return list_


# m2 s1
# 연관된 문장을 찾아오는 함수
async def find_closest_match(client, msg_type, msg_object, user_input, nickname, number):
    final_list = []
    try:
        tokenized_input, n_list = await tokenize_N(user_input)
        n_list = n_list[:number[0]]
        if msg_type == "Buy&Sell":
            for n_word in n_list:
                final_list += await keyword_search(n_word)
            final_list = list(set(final_list))
        
            # final_list.append('---------------keyword------------------')
            final_list += await search_vector(client, nickname, msg_object, number[1]*2, number[1], final_list)
            # final_list.append('---------------object------------------')
            final_list += await search_vector(client, nickname, tokenized_input, number[2]*2, number[2], final_list)
            # final_list.append('---------------tokenized----------------')
            _, indices = index.search(await to_vector(client, user_input), k=number[3])

            extracted_info = await extract_sentences(indices, nickname) 
            n = 0
            while len(final_list) < number[3]:
                if extracted_info[n] not in final_list:
                    final_list.append(extracted_info[n])
                n += 1
        elif msg_type == "Match":
            final_list += await search_vector(client, nickname, msg_object, number[1]*2, number[1], final_list)
            # final_list.append('---------------object------------------')
            final_list += await search_vector(client, nickname, tokenized_input, number[2]*2, number[2], final_list)
            # final_list.append('---------------tokenized----------------')
            _, indices = index.search(await to_vector(client, user_input), k=number[3])
            extracted_info = await extract_sentences(indices, nickname)
            n = 0
            while len(final_list) < number[3]:
                if extracted_info[n] not in final_list:
                    final_list.append(extracted_info[n])
                n += 1
        else:
            for n_word in n_list:
                final_list += await keyword_search(n_word)
            final_list = list(set(final_list))
            # final_list.append('---------------keyword------------------')
            final_list += await search_vector(tokenized_input, number[2]*2, number[2], final_list)
            # final_list.append('---------------tokenized----------------')
            _, indices = index.search(await to_vector(client, user_input), k=number[3])
            extracted_info = await extract_sentences(indices, nickname)
            n = 0
            while len(final_list) < number[3]:
                if extracted_info[n] not in final_list:
                    final_list.append(extracted_info[n])
                n += 1
    except BadRequestError:
        _, indices = index.search(await to_vector(client, user_input), k=number[3])
        final_list = await extract_sentences(indices, nickname)
    # 해당 부분 모든 메시지 다 출력되도록
    except IndexError:
        final_list = await all_data_from_db()
    return final_list


# m2 s2
# 메시지 의도 판단
## 3/22 수정(윤이지) = if문의 입력 대화에서 띄어쓰기 제거
async def classify(msg, matched_agent, client):
    # 활동명만 입력했을 경우 Match나 Buy&Sell로 연결되는 것을 방지하기 위함
    if msg.replace(" ", "") in matched_agent:
        return ["Chat"]
    else:
        prompt = classify_prompt(msg)

        response = client.chat.completions.create(
                        model = "gpt-4-1106-preview",
                        messages=[{"role": "system", "content": prompt}],
                        temperature = 0
                    )
        response = response.choices[0].message.content
        try:
            classification = ast.literal_eval(response)
            if classification[0] == "Buy&Sell":
                return ["Buy&Sell", classification[1]]
            elif classification[0] == "Match":
                return ["Match", classification[1]]
            else:
                return ["Chat"]
        except:
            return ["Chat"]
        

# m2
async def make_profile(client, nickname, method, msg, matched_agent, number):

    if method == 0:
        profile = await find_closest_match(client, '', '', msg, nickname, number)
        return profile
    elif method == 1:
        classification = await classify(msg, matched_agent, client)
        if classification[0] == "Chat":
            return "Chat"
        else:
            profile = await find_closest_match(client, classification[0], classification[1], msg, nickname, number)
            return profile


# if profile == "Chat": ----------------------------------
# m3      
async def chat(client, msg, state):
    chat_prompt = get_chat_prompt(msg)
    state.append({"role": "system", "content": chat_prompt})
    response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=state,
                    stream=False)
    response = response.choices[0].message.content
    result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
    result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
    state.append({"role": "assistant", "content": response})
    return result_ME, result_AI, state

# m4
async def final_process(method, state, matched_agent, matched_msg, user_name):
    sub_matched_agent = re.findall(r'Agent: (.*?)\n', state[-1]['content'])
    for i in range(len(sub_matched_agent)):
        # if sub_matched_agent[i] not in matched_agent:
        matched_agent.append(sub_matched_agent[i])
    if method == 0:
        sub_matched_msg = re.findall(r'메시지: (.*?)$', state[-1]['content'])
        for i in range(len(sub_matched_msg)):
            matched_msg.append(sub_matched_msg[i])
    elif method == 1:
        sub_matched_msg = re.findall(r'메시지: (.*?)\n', state[-1]['content'])
        for i in range(len(sub_matched_msg)):
            matched_msg.append(sub_matched_msg[i])

    matched_agent = matched_agent[-14:]
    matched_msg = matched_msg[-14:]
    del state[0]
    state = state[-9:]
    state_message = {"role": "system",
                 "content": f"""You are a matchmaker, And the user's name is {user_name}.
                 Remember this, Refer to the chatting history below, and Answer the user's message in Korean."""}
    state.insert(0, state_message)
    return state, matched_agent, matched_msg


# m5 s1 ss1
# 매칭 시 Agent만 추출
async def find_agent(msg, profile, client):
    RAG_prompt = matching_rag_prompt(msg, profile)
    response = client.chat.completions.create(
                    model = "gpt-4-1106-preview",
                    messages=[{"role": "system", "content": RAG_prompt}],
                    temperature = 0
                )
    response = response.choices[0].message.content
    return response

# m5 s1 ss2
# 추출된 Agent의 답변형식
def agent_print_form(agent, profile):
    for i in profile:
        if i[0] == agent:
            msg = i[1]
            text = f"""사용자의 요구사항을 바탕으로 다음 에이전트를 추천합니다.\n\nAgent: {agent}\n메시지: {msg}"""
            return text

# m5 s1
async def Matching_print(profile, msg, client):
    print_agent = await find_agent(msg, profile, client)
    if print_agent == 'None':
        final_answer = "죄송합니다, 현재 요청하신 매칭 조건에 적합한 사용자를 찾지 못하였습니다. 원하시는 다른 매칭이 있으신가요?"
    else:
        final_answer = agent_print_form(print_agent, profile)
    return final_answer


# m5
# gpt로 답변 생성하는 경우에는 stream으로 출력가능한 형태 return
# Agent만 추출할 경우 이전 대화이력에 최종답변을 추가한 뒤 return
# input = print_method, 20문장, 메시지, input_prompt

## 3/28 수정
## - method == 1일 경우에도 type='match'시 상품명/가격 json 형태로 return되도록 수정
## - method == 2일 경우 추가
async def printing(client, method, profile, msg, state):
    unmatch_msg = "죄송합니다, 현재 요청하신 매칭 조건에 적합한 사용자를 찾지 못하였습니다. 원하시는 다른 매칭이 있으신가요?"
    if method == 0:
        response =  await Matching_print(profile, msg, client)
        if response == unmatch_msg:
            result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
        else:
            sub_matched_agent = re.findall(r'Agent: (.*?)\n', response)
            sub_matched_msg = re.findall(r'메시지: (.*?)$', response)
            message = await purchase_keyword(sub_matched_msg[0])
            if message == 'no_product':
                result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
            else:
                result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": sub_matched_agent[0], "product_name": message[0], "product_price": message[1], "talk_msg": response}
        state.append({"role": "system", "content": f"매칭리스트: {profile}"})
        state.append({"role": "assistant", "content": response})
        return result_ME, result_AI, state

    elif method == 1:
        state.append({"role": "system", "content": RAG_prompt + f" 사용자: <{msg}>. 매칭리스트: {profile}"})
        response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=state,
                        stream=False)
        response = response.choices[0].message.content
        del state[-1]
        state.append({"role": "system", "content": f"매칭리스트: {profile}"})
        state.append({"role": "assistant", "content": response})
        if response == unmatch_msg:
            result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
        else:
            sub_matched_agent = re.findall(r'Agent: (.*?)\n', response)
            sub_matched_msg = re.findall(r'메시지: (.*?)\n', response)
            message = await purchase_keyword(sub_matched_msg[0])
            if message == 'no_product':
                result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
            else:
                result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
                result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": sub_matched_agent[0], "product_name": message[0], "product_price": message[1], "talk_msg": response}
        return result_ME, result_AI, state
    
    elif method == 2:
        one_agent = await find_agent(msg, profile)
        if response == "None":
            result_ME = { "ai_flag": "M", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'chat', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": unmatch_msg}
            return result_ME, result_AI, state
        for i in profile:
            if i[0] == one_agent:
                agent_msg = i[1]
        state.append({"role": "system", 
                   "content": f"""
                    Quickly match an agent from a provided list with specific criteria. Here's what you need to do:
                    1. {one_agent}: Agent
                    2. {agent_msg}: 메시지
                    3. Answer the <매칭 이유> based on the information above.

                    Format your response as follows:
                    사용자의 요구사항을 바탕으로 다음 에이전트를 추천합니다.

                    Agent: [Recommended Agent]
                    메시지: [Agent's Message]
                    매칭 이유: [Reason for Match]

                    *Cautions:
                    - Only include one recommended agent.
                    - Must answer the reason for matching as concisely as possible in a single sentence. (100 characters less)"""})
        response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=state,
                        stream=False)
        response = response.choices[0].message.content
        del state[-1]
        state.append({"role": "system", "content": f"매칭리스트: {profile}"})
        state.append({"role": "assistant", "content": response})
        sub_matched_agent = re.findall(r'Agent: (.*?)\n', response)
        sub_matched_msg = re.findall(r'메시지: (.*?)\n', response)
        message = await purchase_keyword(sub_matched_msg[0])
        if message == 'no_product':
            result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": response}
        else:
            result_ME = { "ai_flag": "M", "type": 'match', "mem_nick_name": '', "product_name": '', "product_price": 0, "talk_msg": msg}
            result_AI = { "ai_flag": "A", "type": 'match', "mem_nick_name": sub_matched_agent[0], "product_name": message[0], "product_price": message[1], "talk_msg": response}
        return result_ME, result_AI, state