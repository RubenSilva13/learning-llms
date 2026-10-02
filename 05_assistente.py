import os
from datetime import datetime
from dotenv import load_dotenv
from ddgs import DDGS
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain.agents import create_agent

load_dotenv()

reader = PdfReader("documento.pdf")
docs = [
    Document(page_content=p.extract_text() or "", metadata={"pagina": i + 1})
    for i, p in enumerate(reader.pages)
]
pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
retriever = InMemoryVectorStore.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 4})


@tool
def consultar_documento(pergunta: str) -> str:
    """Pesquisa no artigo científico do utilizador sobre o sensor LiDAR em iOS
    (captura de nuvens de pontos, ARKit, fotogrametria). Usa para qualquer
    pergunta sobre o artigo, o trabalho ou o projeto do utilizador."""
    resultados = retriever.invoke(pergunta)
    return "\n\n".join(f"[Página {d.metadata['pagina']}]\n{d.page_content}" for d in resultados)

@tool
def pesquisar_web(consulta: str) -> str:
    """Pesquisa na web informação atual ou local. Usa para factos que não sabes
    e que não estão no artigo do utilizador."""
    resultados = DDGS().text(consulta, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in resultados)

@tool
def hora_atual() -> str:
    """Devolve a data e hora atuais."""
    return datetime.now().strftime("%d/%m/%Y %H:%M")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="conversational",
    temperature=0.2,
    max_new_tokens=1024,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

agente = create_agent(
    model=ChatHuggingFace(llm=llm),
    tools=[consultar_documento, pesquisar_web, hora_atual],
    system_prompt=(
        "És um assistente que responde em português de Portugal.\n"
        "- Perguntas sobre o artigo/trabalho do utilizador: usa consultar_documento e indica as páginas.\n"
        "- Factos atuais ou locais: usa pesquisar_web e indica os links.\n"
        "- Conhecimento geral: responde diretamente.\n"
        "Nunca inventes nomes, números ou referências."
    ),
)


mensagens = []

while True:
    pergunta = input("\nPergunta: ")
    if pergunta.lower() == "sair":
        break
    if not pergunta.strip():
        continue

    mensagens.append({"role": "user", "content": pergunta})
    resultado = agente.invoke({"messages": mensagens})

    for msg in resultado["messages"][len(mensagens):]:
        for chamada in getattr(msg, "tool_calls", []) or []:
            print(f"  🔧 {chamada['name']}({chamada['args']})")

    mensagens = resultado["messages"] 
    print("\n" + mensagens[-1].content)