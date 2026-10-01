# Aprender LLMs com LangChain e Hugging Face

Exercícios práticos de aprendizagem sobre Large Language Models, desde prompts simples até RAG (Retrieval-Augmented Generation), usando LangChain e modelos abertos via Hugging Face Inference Providers.

## Conteúdo

| Ficheiro | Conceito | Descrição |
|---|---|---|
| `01_explicador.py` | Prompt templates e chains | Explica um tema adaptado ao nível (iniciante/avançado) e ao idioma escolhidos |
| `02_chatbot.py` | Memória | Chatbot que se lembra da conversa, com janela das últimas mensagens |
| `03_rag.py` | RAG | Responde a perguntas sobre um PDF, indicando as páginas usadas como fonte |

## Como funciona o RAG

1. O PDF é dividido em pedaços de ~1000 caracteres.
2. Cada pedaço é convertido num embedding (modelo multilingue `paraphrase-multilingual-MiniLM-L12-v2`).
3. Para cada pergunta, são recuperados os 4 pedaços mais semelhantes.
4. O LLM responde usando **apenas** esse contexto — se a resposta não estiver no documento, diz "Não sei" em vez de inventar.

## Instalação

```bash
git clone https://github.com/RubenSilva13/aprender-llms.git
cd aprender-llms
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copia `.env.example` para `.env` e coloca o teu token da Hugging Face (huggingface.co/settings/tokens).

Para o RAG, coloca um PDF na pasta com o nome `documento.pdf`.

## O que aprendi

- Os LLMs **alucinam**: sem dados reais, inventam respostas plausíveis mas falsas.
- A "memória" de um chatbot é apenas reenviar o histórico em cada pedido.
- O RAG reduz alucinações ao obrigar o modelo a responder com base em documentos.
- Tutoriais desatualizam depressa: modelos deixam de estar disponíveis e APIs mudam.

## Tecnologias

Python · LangChain · Hugging Face · Sentence Transformers