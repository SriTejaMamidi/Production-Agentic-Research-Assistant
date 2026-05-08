# Importing the required modules
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tools import search_tool

MAX_RECURSION = 5

# 1. Define the state (what the agent remembers during the conversation)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    recursion_count: int  # tracks how many reasoning loops have occurred

def create_agent():
    # 2. Initializing the local LLM
    llm = ChatOllama(model="llama3.1", temperature=0)

    # 3. Get tools from tools.py and bind them
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools)

    # 4. Assistant node (The Thinking step)
    def chatbot(state: State):
        count = state.get("recursion_count", 0)
        return {
            "messages": [llm_with_tools.invoke(state["messages"])],
            "recursion_count": count + 1
        }

    # 5. Self-reflection node — forces the agent to review its own answer
    # before returning to the user, catching incomplete or looping responses
    def reflect(state: State):
        last_message = state["messages"][-1]
        reflection_prompt = (
            f"Review your previous answer: '{last_message.content}'. "
            "Is it complete and accurate? If yes, confirm. "
            "If not, correct it in one concise response."
        )
        reflection = llm_with_tools.invoke(state["messages"] + [
            {"role": "user", "content": reflection_prompt}
        ])
        return {
            "messages": [reflection],
            "recursion_count": state.get("recursion_count", 0)
        }

    # 6. Recursion guard — routes to reflect or END based on loop count
    def recursion_guard(state: State):
        if state.get("recursion_count", 0) >= MAX_RECURSION:
            return "reflect"
        return tools_condition(state)

    # 7. Build the graph workflow
    workflow = StateGraph(State)

    # 8. Add nodes
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("reflect", reflect)

    # 9. Entry point
    workflow.add_edge(START, "chatbot")

    # 10. Logic: LLM decides to use tools, reflect, or END
    workflow.add_conditional_edges("chatbot", recursion_guard)

    # 11. After using a tool, agent must think again
    workflow.add_edge("tools", "chatbot")

    # 12. After self-reflection, always end
    workflow.add_edge("reflect", END)

    return workflow.compile()
