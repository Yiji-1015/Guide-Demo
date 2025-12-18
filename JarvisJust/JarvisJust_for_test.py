from openai import OpenAI
import pandas as pd
import numpy as np
import faiss
import gradio as gr
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
import re
import time

kiwi = Kiwi(typos="basic", model_type='sbg')
stopwords = Stopwords()
stopwords.remove(('사람', 'NNG'))
# OpenAI api key 입력
nickname = '하렉스'
full_conv_dict = {}

#기존 파일 불러오기
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

df = pd.read_pickle('./20240222_JJAgentData.pkl')
# df['Agent'] = df['Agent'].apply(lambda x: x.replace("\n", ""))
df['Agent'] = df['Agent'].apply(lambda x: re.sub("( |\n)", "", x))
if len(df[df['Agent'] == nickname]) > 0:
    dup_index = df[df['Agent'] == nickname].index
    df.drop(dup_index, axis=0, inplace=True)
else:
    pass

if 'Embeddings' in df.columns:
    vectors_list = df['Embeddings'].tolist()
else:
    df['Embeddings'] = (df['메시지']).apply(lambda x: get_embedding(x))
    vectors_list = df['Embeddings'].tolist()
    df.to_pickle('./2024nnnn_JJAgentData.pkl')

# 매칭이 필요한 대화내용 저장
def update(msg):
    input_vector = get_embedding(msg)
    df.loc[len(df)] = (nickname, msg, input_vector)
    df.to_pickle('./data_for_update.pkl')

# 전체 대화내용 저장
def add_to_conv_dict(diction ,user, msg, full_reply_content):
    if user not in diction:
        diction[user] = {'user':[msg], 'JJ': [full_reply_content]}
    else:
        diction[user]['user'].append(msg)
        diction[user]['JJ'].append(full_reply_content)

def to_vector(x):
    search_vector = get_embedding(x)
    _vector = np.array([search_vector]).astype(np.float32)
    return _vector

def faiss_index(vectors_list):
    vectors = np.array(vectors_list).astype(np.float32)
    #index = faiss.IndexFlatIP(vector_dimension)
    index = faiss.IndexFlatL2(3072)
    #faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

index = faiss_index(vectors_list)

def tokenize_N(text):
    split_s = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form for i in split_s if i.tag == "NNG" or i.tag == "NNP"]
    split_list = [i.form for i in split_s]
    split_f = ','.join(split_list).replace(",", " ")
    return split_f, N_list

def extract_sentences(df, indices, exclude_nickname=None):
    extracted_info = []
    for idx in indices.flatten():
        if idx < len(df):
            agent_name = df.iloc[idx]['Agent']
            message = df.iloc[idx]['메시지']
            # 사용자의 nickname을 포함하는 항목을 제외
            if exclude_nickname is None or agent_name != exclude_nickname:
                extracted_info.append((agent_name, message))
    return extracted_info

# keyword 2개씩 검색해옴
def keyword_search(keyword):
    final_list = []
    k_search = df[df['메시지'].str.contains(keyword)].index
    if len(k_search) > 1:
        n = 0
        while len(final_list) < 2:
            if df.iloc[k_search[n], 0] != nickname:
                final_list += [(df.iloc[k_search[n], 0], df.iloc[k_search[n], 1])]
            n += 1
    elif len(k_search) == 1:
        if df.iloc[k_search[0], 0] != nickname:
            final_list += [(df.iloc[k_search[0], 0], df.iloc[k_search[0], 1])]
        else:
            pass
    else:
        pass
    return final_list

def search_vector(sentence, k1, k2, final_list):
    list_ = []
    _, indices = index.search(to_vector(sentence), k=k1)
    extracted_info = extract_sentences(df, indices, nickname)
    for info in extracted_info[:k2]:
        if info not in final_list:
            list_.append(info)
    return list_

def find_closest_match(user_input, nickname=nickname):
    final_list = []
    try:
        tokenized_input, n_list = tokenize_N(user_input)
        if user_input == 'first_opening_function':
            input2 = to_vector('.')
            _, indices = index.search(input2, k=1)
            final_list += extract_sentences(df, indices, nickname)
            return final_list
        else:
            if len(n_list) < 3:
                for n_word in n_list:
                    final_list += keyword_search(n_word)
            elif len(n_list) >= 3:
                for i in range(2):
                    final_list += keyword_search(n_list[i])
            final_list = list(set(final_list))
            # final_list.append('---------------keyword------------------')
            final_list += search_vector(tokenized_input, 4, 2, final_list)            
            # final_list.append('---------------tokenized----------------')
            _, indices = index.search(to_vector(user_input), k=5)
            extracted_info = extract_sentences(df, indices, nickname)
            n = 0
            while len(final_list) < 10:
                if extracted_info[n] not in final_list:
                    final_list.append(extracted_info[n])
                n += 1
    except:
        _, indices = index.search(to_vector(user_input), k=20)
        final_list = extract_sentences(df, indices, nickname) 
    return final_list

find_closest_match('first_opening_function')

def Matching(msg, profile):
    RAG_prompt = f"""
    * Your task is to perform the following actions:
    1 - <>로 구분된 다음 텍스트에서 사용자의 요구사항을 정확히 파악합니다.
    2 - 사용자의 요구사항을 충족시킬 수 있는 다른 Agent와 매칭합니다. 예를 들어, 연필을 사고 싶어 하는 사람의 경우 연필을 판매하는 Agent와 매칭해야 합니다.
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
    response = client.chat.completions.create(
    model = "gpt-4-1106-preview",
    messages=[{"role": "system", "content": RAG_prompt}],
    temperature = 0
    )
    response = response.choices[0].message.content
    return response

unmatched_msg = "죄송합니다, 현재 요청하신 매칭 조건에 적합한 사용자를 찾지 못하였습니다. 원하시는 다른 매칭이 있으신가요?"

matched_agent = []
def classify(text):
    prompt = f"""If the sentence({text}) seems to buy or sell something, only print 'Buy&Sell'.
                If the sentence seems to be connected or matched someone, only print 'Match'.
                If you cannot judge, only print out 'Match'."""
    response = client.chat.completions.create(
    model = "gpt-4-1106-preview",
    messages=[{"role": "system", "content": prompt}],
    temperature = 0
    )
    response = response.choices[0].message.content
    return response

def print_text(agent):
    text = f"""Agent: {agent}\n메시지: {df.loc[df['Agent'] == agent, '메시지'].iloc[0]}"""
    return text

def kor_eng(text):
    Eng = re.compile("[a-zA-Z]")
    Kor = re.compile("[ㄱ-ㅣ가-힣+]")
    if len(Eng.findall(text))*2/3 > len(Kor.findall(text))*1/3:
        return 'Korean'
        # return 'Must translate all the text to English and answer in English.'
    else:
        # return 'Must translate all the text to Korean and answer in Korean.'
        return 'English'
    
def translate(msg, language):
    prompt = f"""Translate the following text to {language}:{msg} 
                Print out only translated text, and you must observe the form of the text."""
    
    response = client.chat.completions.create(
    model = "gpt-4-1106-preview",
    messages=[{"role": "system", "content": prompt}],
    temperature = 0
    )
    response = response.choices[0].message.content
    return response

def purchase_keyword(msg):
    purchase_prompt = f"""Here is the provided message: {msg}
                    1. provided message가 판매하는 제품에 대해
                    상품명/금액/수량(단위) 3개의 키워드를 추출해 주되, /로 구분해서 키워드만 추출해줘.
                    2. 여러 상품이 있을 경우, 하나의 상품만 출력해 줘.
                    * Cautions
                    1. 상품명에 수량(단위)가 존재할 경우, 이는 제외하고 순수 상품명만 적어줘.
                    2. 상품명이 없는 경우에는 판매 중인 물건을 적어주고, 이것도 없을 경우 'Default'로 표시해줘.
                    3. 금액은 숫자로만 표시해 줘. 다만 금액이 없는 경우 '38000'으로 표시해 줘.
                    4. 수량(단위)가 없는 경우 'Default'으로 표시해 줘.
                    5. 각각 하나씩만 표시해 줘.
                    예시-
                    provided message: 김 선물세트 100g 10,000원에 팝니다.
                    '김 선물세트'/'10000'/'100g'"""
    response = client.chat.completions.create(
    model = "gpt-4-1106-preview",
    messages=[{"role": "system", "content": purchase_prompt}],
    temperature = 0
    )
    response = response.choices[0].message.content
    try:
        if len(response.split('/')) == 3:
            purchase_list = response.split('/')
            if purchase_list[1] == 'Default' or purchase_list[1] == '':
                purchase_list[1] = 38000
                return [purchase_list[0], purchase_list[1], purchase_list[2]]
            elif purchase_list[0] == 'Default':
                return 'no_product'
        # elif response == 'no_product':
        #     return 'no_product'
        return [purchase_list[0], re.sub(r'[^0-9]', '', purchase_list[1]), purchase_list[2]]
    except:
        return 'no_product'
    
def reload_javascript():
    js = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>
    <script>
        function chatConnect() {
            const appConnectDivs = document.querySelectorAll('.chat_connecting:not(.chat_connected)');
            appConnectDivs.forEach(div => {
                const receiverNickname = div.getAttribute('data-agent-name');

                const urlParams = new URLSearchParams(window.location.search);
                const custId = urlParams.get('cust_id');
                const nickname = urlParams.get('nickname');

                const appCall = 'jjcall://openChat/' + JSON.stringify({
                    "s_cust_id": custId,
                    "s_nickname": nickname,
                    "r_cust_id": "",
                    "r_nickname": receiverNickname,
                    "success": "successopenChat0",
                    "error": "erroropenChat0"
                });
                // alert(appCall);
                // console.log(appCall);
                window.location.href = appCall;

                div.classList.add('chat_connected');
            });
        }

        function payConnect() {
            const payConnectDivs = document.querySelectorAll('.pay_connecting:not(.pay_connected)');
            payConnectDivs.forEach(div => {
                const nickname = div.getAttribute('data-agent-name');
                const productName = div.getAttribute('data-product-name');
                const productPrice = div.getAttribute('data-product-price');
                const productQuantity = div.getAttribute('data-product-quantity');

                const paymentCall = 'jjcall://openPaymentResult/' + JSON.stringify({
                    "nickname": nickname,
                    "product_name": decodeURIComponent(productName),
                    "quantity": productQuantity,
                    "amount": productPrice,
                    "success": "successopenPaymentResult0",
                    "error": "erroropenPaymentResult0"
                });
                // alert(paymentCall);
                // console.log(paymentCall);
                window.location.href = paymentCall;

                div.classList.add('pay_connected');
            });
        }

        // MutationObserver
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                if (mutation.addedNodes.length) {
                    chatConnect();
                    payConnect();
                }
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """
    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</html>', f'{js}</html>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

reload_javascript()

custom_button_css = """
    #submit-button {
        background-color: #4CAF50;
        color: green;
    }"""
css = (
    """
#col-container {max-width: 400px; margin-left: auto; margin-right: auto;}
#chatbox {min-height: calc(70vh - 60px);}
.message { font-size: 1.2em; }
footer{display:none !important}

#chatbox {
    border: none !important;
    box-shadow: none !important;
    min-height: calc(100vh - 140px) !important;
}

.wrapper label:first-child {
    display: none !important;
}

.message {
    line-height: normal !important;
}

.message.bot {
    background: #fff !important;
    padding: 1.2rem 1.6rem !important;
    border: #ccc 1px solid !important;
    border-radius: 14px 14px 14px 0px !important;
}

.message.user {
    background: #DE5259 !important;
    color: #fff !important;
    border-color: #DE5259 !important;
    border-radius: 14px 14px 0px 14px !important;
}

#component-4 {
    padding: 20px 0 0 0 !important;
    border-top: #ccc 1px solid !important;
}

.form {
    border: none !important;
    box-shadow: none !important;
}

.wrap label textarea {
    border: none !important;
    border-radius: 100px !important;
    box-shadow: none !important;
    background: #eee !important;
    width: calc(100% - 55px) !important;
    height: auto !important;
    padding: 16px !important;
}

.avatar-container {
    width: 40px !important;
    align-self: flex-end !important;
}

#submit-button {
    position: absolute !important;
    bottom: 0 !important;
    right: 6px !important;
    outline: none !important;
    width: 50px !important;
    height: 50px !important;
    background: url("/resources/icon_send_msg.png") no-repeat center !important;
    background-size: 50px !important;
    overflow: hidden !important;
    text-indent: -999rem !important;
    box-shadow: none !important;
    border: none !important;
}

#component-6 {display: none;}

.gradio-container {padding: 8px 8px !important;}

"""
 + custom_button_css
)

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        state = gr.State([{"role": "assistant", "content":f'안녕하세요 {nickname}님. JarvisJust, JJ입니다. 무엇을 도와드릴까요?'}])
        chatbot = gr.Chatbot(elem_id="chatbox", label="JarvisJust")
        msg = gr.Textbox(show_label=False, placeholder="대화를 입력해주세요", visible=True)
        submit_button = gr.Button("JJ, 연결해줘~", elem_id="submit-button")

        def user(user_message, history):
            return '', history + [[user_message, None]]

        def initial_message(history, state):
            global opening_welcome
            opening_welcome = f"""안녕하세요? JarvisJust(JJ)입니다. 무엇을 도와드릴까요? 

                                찾으시는 내용을 아래처럼 입력해주시면 JJ가 연결해드립니다.
                                예1) 진미채 살 수 있는 곳을 찾아줘
                                예2) 주말에 축구 같이할 사람 찾아줘

                                연결된 이후 상품 구매를 희망하시면,                                        
                                ‘닉네임과 상품 결제해줘’를
                                예) OOO(닉네임) 상품 결제해줘

                                1:1채팅 연결을 원하시면, 
                                ‘닉네임과 연결해줘’를 입력해주세요.
                                예) OOO(닉네임) 연결해줘"""
            state = [state[0]]
            return history + [[None, opening_welcome]], state

        def botv2(msg, history, state):
            msg = history[-1][0]
            if msg == "":
                empty_msg = iter('대화를 입력해주세요.')
                full_reply_content = ''
                for i in range(len('대화를 입력해주세요.')):
                    full_reply_content += next(empty_msg)
                    time.sleep(0.005)
                    yield history + [[None, full_reply_content]]
                return "", history, state
            elif "초기화" in msg or "새로고침" in msg or "다시시작" in msg or "리셋" in msg:
                history = ""
                state = [state[0]]
                yield [[None, opening_welcome]]
                return "", history, state

            state.append({"role":"user", "content":msg})
            language = kor_eng(msg)
            global matched_agent
            for i in range(len(matched_agent)):
                if matched_agent[i] in msg.replace(" ", ""):
                    if classify(msg) == 'Buy&Sell':
                        product_keyword = purchase_keyword(df.loc[df['Agent'] == matched_agent[i], '메시지'].iloc[0])
                        if product_keyword == 'no_product':
                            yield history + [[None, '결제할 상품이 없습니다.']]
                        else:
                            try:
                                pay_nickname = matched_agent[i]
                                pay_product_name =  product_keyword[0]
                                pay_product_price = product_keyword[1]
                                pay_product_quantity = product_keyword[2]
                                pay_msg = f"<div class='pay_connecting' data-agent-name='{pay_nickname}' data-product-name='{pay_product_name}' data-product-price='{pay_product_price}' data-product-quantity='{pay_product_quantity}'>결제를 진행합니다.</div>"
                                # print(pay_msg)
                                yield history + [[None, pay_msg]]
                            except:
                                yield history + [[None, '결제할 상품이 없습니다.']]                   
                        return "", history, state
                    elif classify(msg) == 'Match':
                        connection_msg = f"<div class='chat_connecting' data-agent-name='{matched_agent[i]}'>네 {matched_agent[i]}와 연결해드리겠습니다.</div>"
                        yield history + [[None, connection_msg]]
                    return "", history, state
                else:
                    continue
            profile = find_closest_match(msg)
            print_agent = Matching(msg, profile)
            if print_agent == 'None':
                final_answer = unmatched_msg
            else:
                final_answer = print_text(print_agent)
            if language == 'Korean':
                final_answer = translate(final_answer, language)
            state.append(
                {
                    "role": "system",
                    "content": f"매칭리스트: {profile}",
                }
            )
            state.append(
                {
                    "role": "assistant",
                    "content": f"{final_answer}"
                }
            )
            yield history + [[None, final_answer]]
            update(msg)
            
        def botv3(msg, history, state):
            try:
                global matched_agent
                sub_matched_agent = re.findall(r'Agent: (.*?)\n', history[-1][1])
                for i in range(len(sub_matched_agent)):
                    if sub_matched_agent[i] not in matched_agent:
                        matched_agent.append(sub_matched_agent[i])
            except:
                pass
            state.append({"role": "assistant", "content": history[-1][1]})
            add_to_conv_dict(full_conv_dict, nickname, history[-2][0], history[-1][1])
            matched_agent = matched_agent[-14:]
            state = state[-9:]
            state.insert(0, {"role": "system", "content": f"You are a matchmaker, And the user's name is {nickname}. Remember this, Refer to the chatting history below, and Answer the user's message"})
            return "", history, state

        
        submit_button.click(user, [msg, chatbot], [msg, chatbot]).then(
            botv2, [msg, chatbot, state], [chatbot]
        ).then(botv3, [msg, chatbot, state], [msg, chatbot, state])
        clear = gr.Button("새로운 대화 시작")
        clear.click(lambda: None, None, chatbot, queue=False).then(
            initial_message, inputs=[chatbot, state], outputs=[chatbot, state]
        )
        demo.load(initial_message, inputs=[chatbot, state], outputs=[chatbot, state])

demo.launch(
    share=True,
    server_name = '0.0.0.0',
    height='800px',
    width='500px',
    debug=True,
    show_error=False,
    server_port=8401
)