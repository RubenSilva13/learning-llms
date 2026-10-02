import os
from datetime import datetime
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import create_agent

load_dotenv()

@tool
def hora_atual() -> str:
    """Retorna a hora atual."""
    return datetime.now().strftime("%d/%m/%Y %H:%M")

@tool
def pesquisar_web(consulta: str) -> str:
    """Pesquisa na web informaçao atual ou local. Usar para factos que nao sei."""
    resultados = DDGS().text(consulta, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in resultados)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task = "conversational",
    temperature=0.2,
    max_new_tokens=1024,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm)

agent = create_agent(
    model=chat,
    tools=[hora_atual, pesquisar_web],
    system_prompt=(
        "Es um assistente que responde em portugues de Portugal"
        "Para factos atuais ou locais, usa sempre a pesquisa na web e indica as fontes."
        "Nunca investes nomes, moradas ou numeros de telefone. Se nao souberes a resposta, diz 'Nao sei'."
    ),
)

while True:
    pergunta = input("\nPergunta: ")
    if pergunta.lower() == "sair":
        break
    resultado = agent.invoke({"messages": [{"role": "user", "content": pergunta}]})


    for msg in resultado["messages"]:
        for chamada in getattr(msg, "tool_calls", []) or []:
            print(f" {chamada['name']} ({chamada['args']})")
    print("\n" + resultado["messages"][-1].content)
