from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimension=300)

documents = [
    "Python is a high-level programming language widely used for web development, data science, and automation.",
    "Java is an object-oriented programming language known for its platform independence using the JVM.",
    "JavaScript is primarily used to create interactive websites and runs in web browsers.",
    "C++ is a powerful programming language commonly used for game development and system programming.",
    "Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data.",
    "Deep Learning uses neural networks with multiple layers to solve complex AI problems.",
    "Natural Language Processing (NLP) enables computers to understand and generate human language.",
    "LangChain is a framework for building applications powered by Large Language Models (LLMs).",
    "Vector databases store embeddings and enable semantic search using similarity matching.",
    "FAISS is an open-source library developed by Meta for efficient similarity search over vector embeddings."
]

query =  "Which framework helps build applications using large language models?"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)
print("Similarities:", similarities)