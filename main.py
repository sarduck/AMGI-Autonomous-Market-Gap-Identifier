import os
import chromadb
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM

# Import our new Harvester functions from data_ingestion.py
from data_ingestion import get_youtube_video_ids, fetch_youtube_transcripts

# ==========================================
# 1. SETUP & DYNAMIC INPUT
# ==========================================
os.environ["TAVILY_API_KEY"] = "put key here"

print("==========================================")
print("🎯 MULTI-AGENT MARKET GAP IDENTIFIER")
print("==========================================")

# Gather exact parameters from the user
industry = input("1. Enter the target industry or product: ")
location = input("2. Enter a specific city/region (Press Enter for global): ")

# Dynamically build the search query
if location.strip():
    search_query = f"biggest user complaints and frustrations with {industry} in {location}"
else:
    search_query = f"biggest user complaints and frustrations with {industry}"

print("\nConnecting to local LLM (gpt-oss:20b)...")
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434"
)

# Initialize Tavily
tavily_engine = TavilySearchResults()

@tool("Competitor Web Search")
def search_tool(query: str) -> str:
    """Searches the live web for companies, competitors, and market data."""
    return tavily_engine.invoke({"query": query})

# ==========================================
# 2. THE HARVESTER & MEMORY (TRUE RAG)
# ==========================================
print(f"\n🔍 Harvester initializing... Searching for: '{search_query}'")

# 1. Fetch the data dynamically
vids = get_youtube_video_ids(search_query, limit=3)
chunks = fetch_youtube_transcripts(vids)

# THE FIX: Add a safety net in case the Harvester finds nothing
if not chunks:
    print("\n❌ CRITICAL ERROR: The Harvester could not find any valid transcripts for this query.")
    print("This usually means YouTube had no videos with closed captions for this topic. Exiting...")
    exit()

print(f"\nStoring {len(chunks)} data chunks in ChromaDB...")
chroma_client = chromadb.Client()

# Reset collection for fresh runs so old data doesn't mix with new data
try:
    chroma_client.delete_collection(name="market_research")
except Exception:
    pass
collection = chroma_client.create_collection(name="market_research")

# Add the chunked paragraphs to the database
collection.add(
    documents=chunks,
    metadatas=[{"source": f"youtube_{i}"} for i in range(len(chunks))],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# 2. TRUE RAG: Ask ChromaDB to filter out the fluff and only return complaints
print("🧠 Extracting high-density complaint data via Semantic Search...")
rag_results = collection.query(
    query_texts=["frustrations, bugs, worst features, negative reviews, I hate it when"],
    n_results=5 # Only grab the top 5 most relevant paragraphs
)

# Combine those top 5 chunks into one dense string for the Data Scout
focused_context = "\n\n".join(rag_results['documents'][0])

# ==========================================
# 3. DEFINE THE AGENTS
# ==========================================
print("\nInitializing Agents...")

scout = Agent(
    role='Data Scout',
    goal='Read the provided context and extract exactly 3 specific user complaints.',
    backstory='You are a strict data analyst. You only report facts. You must back up every claim with a direct quote.',
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
    goal=f'Search the web for companies that already do what the Architect proposed in the {industry} space.',
    backstory='You are a ruthless market researcher. You use the web search tool to find existing competitors for the proposed idea.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

critic = Agent(
    role='Final Evaluator',
    goal='Review the product idea and competitors. Give a final GREEN LIGHT or RED LIGHT verdict.',
    backstory='You are a harsh venture capitalist. If competitors exist, you give a RED LIGHT. If the idea is unique, GREEN LIGHT. You MUST include the exact source quotes in your final report.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ==========================================
# 4. DEFINE THE TASKS (With Strict Citations)
# ==========================================
task1 = Task(
    description=f"Analyze this dense complaint data:\n{focused_context}\n\nList exactly 3 user complaints. For every complaint, you MUST include a direct, word-for-word quote from the text that proves the complaint is real.",
    expected_output="A bulleted list of 3 complaints. Each bullet must have a sub-bullet titled 'Evidence:' containing the exact quote.",
    agent=scout
)

task2 = Task(
    description="Take the Scout's 3 complaints and propose exactly 1 realistic product idea to solve them.",
    expected_output="A short paragraph describing the proposed product.",
    agent=architect
)

task3 = Task(
    description=f"Take the Architect's product idea and use your search tool to find at least 2 existing competitors in the {industry} market.",
    expected_output="A list of existing competitors and what they do.",
    agent=analyst
)

task4 = Task(
    description="Review the proposed idea and the competitor list. Write a final 'GREEN LIGHT' or 'RED LIGHT' report. You MUST include the direct quotes found by the Data Scout in your final report to prove the market gap is based on real user data.",
    expected_output="A final report starting with GREEN LIGHT or RED LIGHT, followed by the product analysis, the competitor list, and a 'Source Evidence' section containing the exact quotes.",
    agent=critic
)

# ==========================================
# 5. EXECUTE THE PIPELINE
# ==========================================
market_crew = Crew(
    agents=[scout, architect, analyst, critic],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential 
)

print("\n🚀 Kicking off the multi-agent system...\n")
result = market_crew.kickoff()

print("\n==========================================")
print("🎯 FINAL SYSTEM OUTPUT")
print("==========================================")
print(result)