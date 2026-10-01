import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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
     "És um professor de informática.\n"
     "Responde SEMPRE no idioma indicado pelo utilizador.\n"
     "Nível 'iniciante': linguagem simples, sem termos técnicos, usa uma analogia do dia a dia. Máximo 5 frases.\n"
     "Nível 'avançado': usa terminologia técnica, explica como funciona por dentro e dá um exemplo concreto."),
    ("human", "Tema: {tema}\nNível: {nivel}\nIdioma: {idioma}"),
])

chain = prompt | chat | StrOutputParser()

while True:
    tema = input("\nTema (ou 'sair' para terminar): ")
    if tema.lower() == "sair":
        break

    nivel = input("Nivel (iniciante/avançado): ")

    idioma = input("indica o idioma: ")

    resposta = chain.invoke({"tema": tema, "nivel": nivel, "idioma": idioma})
    print("\n" + resposta)
