import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load env
load_dotenv()

def test_tavily():
    print("\n--- Testing Tavily ---")
    try:
        from langgraph.prebuilt import ToolExecutor
        from src.tools.search import web_search
        res = web_search.invoke("Test query")
        print(f"✅ Tavily Success: Found {len(res)} results")
    except Exception as e:
        print(f"❌ Tavily Failed: {e}")

def test_gemini():
    print("\n--- Testing Gemini (Research) ---")
    try:
        from src.utils.llm_factory import get_research_llm
        llm = get_research_llm()
        msg = llm.invoke([HumanMessage(content="Hello from Gemini")])
        print(f"✅ Gemini Success: {msg.content[:50]}...")
    except Exception as e:
        print(f"❌ Gemini Failed: {e}")

def test_bedrock():
    print("\n--- Testing Bedrock (Analysis/Report) ---")
    try:
        from src.utils.llm_factory import get_analysis_llm
        llm = get_analysis_llm()
        # Verify it's Bedrock
        print(f"Type: {type(llm)}")
        msg = llm.invoke([HumanMessage(content="Hello from Bedrock")])
        print(f"✅ Bedrock Success: {msg.content[:50]}...")
    except Exception as e:
        print(f"❌ Bedrock Failed: {e}")

if __name__ == "__main__":
    test_tavily()
    test_gemini()
    test_bedrock()
