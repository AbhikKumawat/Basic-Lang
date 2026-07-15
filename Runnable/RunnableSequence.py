from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

Template1 = PromptTemplate(
    template='Write a song about following {topic}',
    input_variables=["topic"],
)

Template2 = PromptTemplate(
    template='Explain following {text}',
    input_variables=["text"]
)

chain = RunnableSequence(Template1, model, parser, Template2, model, parser)

print(chain.invoke({'topic':'Black Magic'}))