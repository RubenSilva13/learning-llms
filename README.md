# Aprender LLMs com LangChain e Hugging Face

Exercícios práticos de aprendizagem sobre Large Language Models, desde prompts simples até RAG (Retrieval-Augmented Generation), usando LangChain e modelos abertos via Hugging Face Inference Providers.

## Conteúdo

| Ficheiro | Conceito | Descrição |
|---|---|---|
| `01_explicador.py` | Prompt templates e chains | Explica um tema adaptado ao nível (iniciante/avançado) e ao idioma escolhidos |
| `02_chatbot.py` | Memória | Chatbot que se lembra da conversa, com janela das últimas mensagens |
| `03_rag.py` | RAG | Responde a perguntas sobre um PDF, indicando as páginas usadas como fonte |
| `04_agente.py` | Agentes e ferramentas | Agente que decide quando usar ferramentas (hora atual, pesquisa web) e indica as fontes |
| `05_assistente.py` | Agente + RAG + memória | Assistente que decide entre consultar o PDF, pesquisar na web ou responder diretamente, mantendo o contexto da conversa |

## Como funciona o RAG

1. O PDF é dividido em pedaços de ~1000 caracteres.
2. Cada pedaço é convertido num embedding (modelo multilingue `paraphrase-multilingual-MiniLM-L12-v2`).
3. Para cada pergunta, são recuperados os 4 pedaços mais semelhantes.
4. O LLM responde usando apenas esse contexto — se a resposta não estiver no documento, diz "Não sei" em vez de inventar.

## Instalação

```bash
git clone https://github.com/RubenSilva13/learning-llms.git
cd learning-llms
python -m venv venv
venv\Scripts\activate        
pip install -r requirements.txt
```

Copia `.env.example` para `.env` e coloca o teu token da Hugging Face (huggingface.co/settings/tokens).

Para o RAG, coloca um PDF na pasta com o nome `documento.pdf`.

## O que aprendi

- Os LLMs alucinam: sem dados reais, inventam respostas plausíveis mas falsas.
- A "memória" de um chatbot é apenas reenviar o histórico em cada pedido.
- O RAG reduz alucinações ao obrigar o modelo a responder com base em documentos.
- Tutoriais desatualizam depressa: modelos deixam de estar disponíveis e APIs mudam.
- Um agente é um LLM que decide que ferramentas usar; a docstring de cada ferramenta é o que o guia.
- Com pesquisa web, o agente responde com dados reais — mas a resposta só é tão boa quanto as fontes.
- Aprendi que, dependendo de como a pergunta é formulada, a resposta pode ser diferente. Quando questionado com uma pergunta mais ambígua, o modelo disse que o dispositivo que utilizei nos testes foi o iPhone 12 Pro, o que é falso. Com uma pergunta mais específica, detetou o verdadeiro dispositivo, o iPhone 14 Pro (página 6).
  - Causa: o retriever devolveu os pedaços do estado da arte, onde cito um estudo que usou o iPhone 12 Pro, e o modelo atribuiu esse dispositivo ao meu trabalho. O modelo só vê os pedaços que o retriever encontra, não o documento inteiro. Uma resposta bem estruturada e convincente não é necessariamente correta, é essencial indicar e verificar as fontes.
  
## Tecnologias

Python · LangChain · Hugging Face · Sentence Transformers