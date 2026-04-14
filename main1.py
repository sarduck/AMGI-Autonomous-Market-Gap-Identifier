import os
import chromadb
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai.tools import tool

# THE FIX 1: Import the native LLM class directly from crewai
from crewai import Agent, Task, Crew, Process, LLM

# Import our working YouTube function
from data_ingestion1 import fetch_youtube_data

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Add your Tavily API Key here
os.environ["TAVILY_API_KEY"] = "put key here"

# Connect to your local Ollama instance
print("Connecting to local LLM...")

# THE FIX 2: Use the new LLM syntax with the "ollama/" prefix
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434"
)

# Initialize Tavily Web Search Tool
tavily_engine = TavilySearchResults()

@tool("Competitor Web Search")
def search_tool(search_query: str) -> str:
    """Searches the live web for companies, competitors, and market data."""
    return tavily_engine.invoke({"query": search_query})

# ==========================================
# 2. DATA INGESTION & MEMORY (CHROMADB)
# ==========================================
print("Fetching and cleaning YouTube data...")

# THE FIX 3: Update the video ID to the working one!
video_data = fetch_youtube_data("eFUB_jL_XcM") 

print("Storing data in ChromaDB...")
# Create a local database
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="market_research")

# Add our cleaned text to the database
collection.add(
    documents=[video_data],
    metadatas=[{"source": "youtube"}],
    ids=["doc1"]
)

# ==========================================
# 3. DEFINE THE AGENTS
# ==========================================
print("Initializing Agents...")

scout = Agent(
    role='Data Scout',
    goal='Read the provided text and list exactly 3 specific user complaints or unmet needs.',
    backstory='You are a strict data analyst. You only report facts found in the provided database text.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

architect = Agent(
    role='Solution Architect',
    goal='Create 1 specific product idea that solves the 3 complaints identified by the Scout.',
    backstory='You are a pragmatic product manager. Your solutions must be realistic and directly address the user complaints.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

analyst = Agent(
    role='Competitor Analyst',
    goal='Search the web for companies that already do what the Architect proposed.',
    backstory='You are a ruthless market researcher. You use the web search tool to find existing competitors for the proposed idea.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

critic = Agent(
    role='Final Evaluator',
    goal='Review the product idea and the competitors. Give a final GREEN LIGHT or RED LIGHT verdict.',
    backstory='You are a harsh venture capitalist. If competitors exist, you give a RED LIGHT. If the idea is unique and solves the complaints, you give a GREEN LIGHT. Output a final short report.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ==========================================
# 4. DEFINE THE TASKS (The Linear Flow)
# ==========================================
task1 = Task(
    description=f"Analyze this raw database text and list 3 user complaints: {video_data}",
    expected_output="A bulleted list of 3 specific complaints.",
    agent=scout
)

task2 = Task(
    description="Take the Scout's 3 complaints and propose exactly 1 realistic product idea to solve them.",
    expected_output="A short paragraph describing the proposed product.",
    agent=architect
)

task3 = Task(
    description="Take the Architect's product idea and use your search tool to find at least 2 existing competitors.",
    expected_output="A list of existing competitors and what they do.",
    agent=analyst
)

task4 = Task(
    description="Review the proposed idea and the competitor list. Write a final 'GREEN LIGHT' or 'RED LIGHT' report.",
    expected_output="A final 3-sentence report starting with GREEN LIGHT or RED LIGHT.",
    agent=critic
)

# ==========================================
# 5. EXECUTE THE PIPELINE
# ==========================================
market_crew = Crew(
    agents=[scout, architect, analyst, critic],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential # Enforces the strict linear MVP flow
)

print("\n🚀 Kicking off the multi-agent system...\n")
result = market_crew.kickoff()

print("\n==========================================")
print("🎯 FINAL SYSTEM OUTPUT")
print("==========================================")
print(result)