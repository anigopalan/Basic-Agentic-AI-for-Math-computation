from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOllama

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

add_tool = Tool(
    name="add",
    func=add,
    description="Adds two numbers"
)

multiply_tool = Tool(
    name="multiply",
    func=multiply,
    description="Multiply two numbers"
)

llm = ChatOllama(model="llama3")
agent = initialize_agent(
    tools=[add_tool, multiply_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    result = agent.run("Add 25 and 15, then multiply the result by 2")
    print("\nFinal Answer:", result)
