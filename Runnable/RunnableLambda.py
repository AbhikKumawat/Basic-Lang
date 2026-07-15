from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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
    template="Write a short poem about {topic}",
    input_variables=["topic"],
)


translate = RunnableLambda(
    lambda x: {"topic": f"{x['topic']} (translated to English)"}
)

chain = (
    translate| prompt| model| parser
)

result = chain.invoke({"topic": "Kala Jadu"})
print(result)