from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a short song about {topic}",
    input_variables=["topic"],
)

song_chain = prompt | model | parser

chain = RunnablePassthrough.assign(
    song=song_chain
)

result = chain.invoke({"topic": "Black Magic"})

print(result)