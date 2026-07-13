import langchain 
from dotenv   import load_dotenv
print(langchain.__version__)

import os

load_dotenv()

print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))