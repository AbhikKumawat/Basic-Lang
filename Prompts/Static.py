import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)


chat = ChatHuggingFace(llm=llm)


st.set_page_config(page_title="AI Assistant")

st.title("AI Assistant")

user_input = st.text_area("Ask anything")


if st.button("Submit"):

    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            response = chat.invoke(user_input)
            st.success(response.content)

        except Exception as e:
            st.error(f"Error: {e}")