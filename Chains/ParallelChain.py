from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel       # lets you execute multiple operations concurrently 
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")\

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short notes for following {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Make a quiz for following {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz'])

parser = StrOutputParser()

ParallelChain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

chain = ParallelChain | merge_chain

text = 'Generative Artificial Intelligence, or GenAI, represents a groundbreaking shift in technology where machines evolve from analytical tools into digital creators. Unlike traditional artificial intelligence that focuses purely on analyzing existing data, recognizing patterns, or classifying information, GenAI utilizes massive machine learning models to synthesize entirely new and original content. By studying vast datasets of human creation, these advanced models learn the deep structural rules of language, art, audio, and programming. Consequently, when a user inputs a simple, conversational text prompt, the AI uses statistical probability to predict and construct a logical output from scratch. This technology powers well-known platforms like Large Language Models for writing text and debugging code, as well as diffusion models for rendering high-quality digital artwork and video. Today, GenAI acts as a massive productivity multiplier across industries, helping users brainstorm concepts, draft business materials, and automate repetitive tasks to bypass the "blank page" problem. However, because these systems rely on pattern replication rather than factual understanding or human consciousness, they can occasionally generate believable but entirely fabricated errors known as hallucinations, meaning human oversight remains essential.'

result = chain.invoke({'text': text})
print(result)