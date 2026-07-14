##Structure Output Parsers is removed from latest version langchain so it won't work so use old version.##

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

# Define the expected fields
response_schemas = [
    ResponseSchema(
        name="summary",
        description="A concise summary of the given topic"
    )
]

# Create the parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create the prompt
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

# Create the chain
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"topic": "football"})

print(result)
print(type(result))