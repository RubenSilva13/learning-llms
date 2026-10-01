import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

docs = PyPDFLoader("documento.pdf").load()
print(f"Páginas : {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pedacos = splitter.split_documents(docs)
print(f"Pedacos : {len(pedacos)}")

embeddings = HuggingFaceEmbeddings (
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

)
vectorstore = InMemoryVectorStore.from_documents(pedacos, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="conversational",
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Responde em portugues de portugal, usando apenas o contexto abaixo.\n"
     "Se a respota não estiver no contexto, responde 'Não sei'.\n"
     "Contexto: \n{contexto}"),
    ("human", "{pergunta}"),
])

def juntar(pedacos):
    return "\n\n".join(p.page_content for p in pedacos)

chain = (
    {"contexto": retriever | juntar, "pergunta": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

while True:
    pergunta = input("\nPergunta: ")
    if pergunta.lower() == "sair":
        break

    print("\n" + chain.invoke(pergunta))

    fontes = retriever.invoke(pergunta)
    paginas = sorted({p.metadata["page"] + 1 for p in fontes})
    print(f"\n[Fontes: páginas {paginas}]")


