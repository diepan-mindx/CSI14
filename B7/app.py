import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import streamlit as st

# =====================================
# setup api 
# =====================================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
# print(google_api_key)
client = genai.Client(api_key=google_api_key)

# =====================================
# load menu 
# =====================================
menu_df = pd.read_csv("menu.csv", index_col=[0])

# =====================================
# tao LLM
# =====================================
MODEL_VERSION = "gemini-2.5-flash"
SYSTEM_INTRO = f"""
                Bạn tên là PhoBot, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng đến nhà hàng Viet Cuisine.
                Các chức năng mà bạn hỗ trợ gồm:
                1. Giới thiệu nhà hàng Viet Cuisine: là một nhà hàng thành lập bởi người Việt, ở địa chỉ 329 Scottmouth, Georgia, USA
                2. Giới thiệu menu của nhà hàng, gồm các món: {', '.join(menu_df["name"].to_list())}.
                3. Lịch mở cửa của nhà hàng: từ T2 -> T6 sẽ hoạt động từ 9:30 sáng tới 8:30 tối, T7 + CN hoạt động từ 8:30 sáng tới 10:00 tối. 
                Ngoài các chức năng trên, bạn không hỗ trợ chức năng nào khác. Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. Xin liên hệ nhân viên nhà hàng qua hotline 318-237-3870 để được trợ giúp.'
                Hãy có thái độ thân thiện và lịch sự khi nói chuyện với khác hàng, vì khách hàng là thượng đế.
                """
# model = client.models.generate_content(
#                         model=MODEL_VERSION,
#                         contents="",
#                         config=types.GenerateContentConfig(
#                                 system_instruction=SYSTEM_INTRO
#                                 )
#                         )

# =====================================
# load cau noi khi mo dau LLM
# =====================================
with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
    # tao function de goi den
    functions = config.get('functions', 'giới thiệu nhà hàng')
    initial_bot_message = config.get('initial_bot_message', 'Xin chào! Bạn cần hỗ trợ gì?')
    
# =====================================
# Ham tro chuyen voi chatbot
# =====================================
def restaurant_chatbot():
    st.title("Trợ lý ảo cho nhà hàng")
    st.write("Xin chào! Tôi là trợ lý online của nhà hàng Viet Cuisine. Bạn cần giúp đỡ gì?")
    st.write("(Bạn có thể hỏi tôi về thời gian mở cửa, menu món ăn, ...)")
    
    # nếu chưa có lịch sử trò chuyện
    if 'conversation_log' not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": initial_bot_message}
        ]
        
    # neu da co lich su tro chuyen, hien thi lich su ra man hinh
    for message in st.session_state.conversation_log:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.write(message['content'])
                
                
    # user input (prompt)
    if prompt := st.chat_input("Nhập yêu cầu của bạn tại đây ..."):
        # hien thi prompt cua nguoi dung
        with st.chat_message("user"):
            st.write(prompt)
        # them vao log
        st.session_state.conversation_log.append({'role': 'user', 'content': prompt})
    
        # kiem tra prompt co de cap den meu (ki thuat cat chu)
        response = ""
        bot_reply = ""
        if "menu" in prompt.lower() or "món" in prompt.lower():
            bot_reply = '\n\n'.join([f"**{row['name']}**: {row['description']}"  for ind, row in menu_df.iterrows()])
        else:
            # tao llm
            response = client.models.generate_content(
                        model=MODEL_VERSION,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INTRO
                                )
                        ) 
            bot_reply = response.text
        
        # hien thi cau tra loi tu llm
        with st.chat_message("assistant"):
            st.write(bot_reply)
        # them vao log
        st.session_state.conversation_log.append({'role': 'assistant', 'content': bot_reply})
        
        
# hien thi ra man hinh
if __name__ == "__main__":
    restaurant_chatbot()
        