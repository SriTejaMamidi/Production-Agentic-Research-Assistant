#Importing the required modules
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from agent import create_agent
def main():
    # 1. Page Configuration
    st.set_page_config(page_title="Agentic Researcher", page_icon="🤖")
    st.title("🤖 AI Agentic Researcher")
    st.markdown("""
    This agent uses **LangGraph** and **Llama3.1** to browse the web and reason through complex questions.
    It includes a **self-reflection node** and **recursion guard** to prevent infinite reasoning loops.
    """)
    # 2. Initialize the agent and chat history in session state
    if "agent" not in st.session_state:
        with st.spinner("Initializing Agentic Brain..."):
            st.session_state.agent = create_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. Display the chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # 4. Handle User Input
    if prompt := st.chat_input("What would you like me to research today?"):
        # Adds user message to the history
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generating the AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking & Searching..."):
                # Pass messages to the LangGraph compiled agent
                inputs = {
                    "messages": st.session_state.messages,
                    "recursion_count": 0  # reset counter for each new user query
                }
                result = st.session_state.agent.invoke(inputs)

                # The last message in the list is the final answer
                final_response = result["messages"][-1].content
                st.markdown(final_response)

        # Adding AI response to history
        st.session_state.messages.append(AIMessage(content=final_response))


if __name__ == "__main__":
    main()
