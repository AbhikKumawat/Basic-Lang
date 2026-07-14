# Basic-Lang

Basic-Lang is a beginner-friendly repository for learning the fundamentals of **LangChain**, **Prompt Engineering**, and **LLM-based applications** using Python. It contains simple examples covering chat models, embeddings, memory, structured outputs, and output parsers.

## Project Structure

```text
Basic-Lang/
├── ChatModels/          # Chat model examples
├── EmbeddedModels/      # Embedding model examples
├── LLM's/               # LLM examples
├── Prompts/             # Prompt templates and engineering
├── Memory.py            # Memory implementation
├── OutputParsers.py     # Output parser examples
├── StructuredOutput.py  # Structured output examples
├── test.py              # Testing script
├── requirements.txt     # Project dependencies
└── README.md
```

## Features

- LangChain Basics
- Prompt Engineering
- Chat Models
- LLM Integration
- Embedding Models
- Memory
- Output Parsers
- Structured Outputs

## Installation

```bash
git clone https://github.com/AbhikKumawat/Basic-Lang.git
cd Basic-Lang

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## Run Examples

```bash
python Memory.py
python OutputParsers.py
python StructuredOutput.py
```

## Tech Stack

- Python
- LangChain
- OpenAI
- Hugging Face
- Pydantic

## Contributing

Contributions are welcome. Feel free to fork the repository, make improvements, and submit a pull request.

