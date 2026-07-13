# LG_Model

LG_Model is a beginner-friendly project that demonstrates the basics of **LangChain**, **Prompt Engineering**, and **Hugging Face** using Python and Streamlit.

## Features

- Chat Models
- Static Prompt
- Dynamic Prompt
- Embedding Models
- Streamlit Interface
- Hugging Face Integration
- Memory

## Project Structure

```
LG_Model/
├── ChatModels/
├── EmbeddedModels/
├── LLM's/
├── Prompts/
|-- Memory.py
├── .env
├── requirements.txt
└── test.py
```

## Installation

```bash
git clone <repository-url>
cd LG_Model

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_api_key
```

## Run

Static Prompt

```bash
streamlit run Prompts/Static.py
```

Dynamic Prompt

```bash
streamlit run Prompts/Dynamic.py
```

## Technologies

- Python
- LangChain
- Hugging Face
- Streamlit
- Prompt Engineering

## Author

**Abhik Kumawat**
