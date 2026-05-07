from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

def get_tools():
    """Defines the tools available to the agent.
    We start with DuckDuckGo for free, live web searching."""
    return [search_tool]
