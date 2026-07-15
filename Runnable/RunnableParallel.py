from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
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

# Prompt 1
song_prompt = PromptTemplate(
    template="Write a song about {topic}",
    input_variables=["topic"]
)

# Prompt 2
poem_prompt = PromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

# Two independent chains
song_chain = song_prompt | model | parser
poem_chain = poem_prompt | model | parser

# Execute both in parallel
parallel_chain = RunnableParallel(
    song=song_chain,
    poem=poem_chain
)

result = parallel_chain.invoke({"topic": "Black Magic"})

print(result)