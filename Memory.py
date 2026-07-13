from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

chat = ChatHuggingFace(llm=llm)

# Store conversation
messages = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Save user's message
    messages.append(HumanMessage(content=user_input))

    # Get AI response
    response = chat.invoke(messages)
    print("AI:", response.content)

    # Save AI's reply
    messages.append(AIMessage(content=response.content))