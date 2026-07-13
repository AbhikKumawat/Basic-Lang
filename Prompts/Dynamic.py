import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.5,
)

chat = ChatHuggingFace(llm=llm)

# Dynamic Prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are an expert AI assistant.

Explain the following topic in simple language.

Topic: {topic}

Answer:
"""
)

st.title("AI Assistant-D")

user_input = st.text_input("Enter a topic")

if st.button("Submit"):

    final_prompt = prompt.format(topic=user_input)

    response = chat.invoke(final_prompt)

    st.write(response.content)