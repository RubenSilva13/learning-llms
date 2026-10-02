import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="conversational",
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

chat = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages ([
    ("system", "És um assistente simpatico. Respondes em Portugues de Portugal, de forma curta."),
    MessagesPlaceholder(variable_name="historico"),
    ("human", "{pergunta}"),
])

chain = prompt | chat | StrOutputParser()

historico = []

while True:
    pergunta = input("\nTu:")
    if pergunta.lower() == "sair":
        break   

    resposta = chain.invoke({"historico": historico, "pergunta": pergunta})
    print("Bot: ", resposta)

    historico.append(HumanMessage(content=pergunta))
    historico.append(AIMessage(content=resposta))
