from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# Define the multiply tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# Initialize the LLM with tools
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([multiply])

# Tool calling node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Tool execution node
def execute_tools(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return {"messages": [last_message]}
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "multiply":
            result = multiply(**tool_call["args"])
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )
    
    return {"messages": tool_results}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("execute_tools", execute_tools)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", "execute_tools")
builder.add_edge("execute_tools", END)
graph = builder.compile()

# Run the chain
messages = graph.invoke({"messages": [HumanMessage(content="Multiply 2 and 3")]})

# Print results
for m in messages['messages']:
    print(f"Message type: {type(m).__name__}")
    print(f"Content: {m.content}")
    print("---") 