from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)


# Define the output structure
class Summary(BaseModel):
    summary: str = Field(description="A concise summary of the given topic")


# Create the parser
parser = PydanticOutputParser(pydantic_object=Summary)


# Prompt
prompt = PromptTemplate(
    template="""
Summarize the following topic.

Topic: {topic}

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)


# Chain
chain = prompt | model | parser

# Invoke
result = chain.invoke({"topic": "football"})

print(result)
print(type(result))
print(result.summary)