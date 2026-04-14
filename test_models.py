import os
import chromadb
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM

# ==========================================
# 1. THE FIXED INPUT DATA (The Control Variable)
# ==========================================
# This text contains 3 specific, highly relatable consumer complaints.
fixed_complaint_data = """
I used to order from delivery apps all the time, but the experience has gotten terrible lately. 
First, the hidden fees are completely out of control. A $15 burger ends up costing $28 at checkout after adding the service fee, delivery fee, and a 'small cart' penalty, making it unaffordable for an everyday lunch. 
Second, the food arrives cold and soggy half the time. This happens because the app forces drivers to 'stack' multiple orders, meaning my food sits in a car while the driver stops at two other houses first. 
Lastly, the delivery hand-off in large apartment complexes is a nightmare. The app's GPS pin always drops at the main leasing office instead of my specific building, so the drivers get frustrated, give up, and just leave my food outside in the rain.
"""

# ==========================================
# 2. CHOOSE YOUR ENGINE (The Independent Variable)
# ==========================================
# TO TEST MODEL A: Uncomment the 8B model
# engine_model = "ollama/llama3.1:8b"

# TO TEST MODEL B: Uncomment the 20B model
engine_model = "ollama/gpt-oss:20b"

print(f"Loading Test Environment with engine: {engine_model}...")

llm = LLM(
    model=engine_model,
    base_url="http://localhost:11434"
)

# Mocking the search tool to match the new food delivery industry
@tool("Competitor Web Search")
def search_tool(query: str) -> str:
    """MOCK TOOL: Simulates a web search for testing purposes."""
    return f"Search results for {query}: UberEats, DoorDash, and Grubhub dominate the market. They all rely on gig-economy drivers with personal cars, use stacked orders for efficiency, and charge high service fees to both restaurants and consumers."

# ==========================================
# 3. DIRECT MEMORY INJECTION
# ==========================================
print("Injecting fixed control data into ChromaDB...")
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection(name="model_test")
except Exception:
    pass
collection = chroma_client.create_collection(name="model_test")

collection.add(
    documents=[fixed_complaint_data],
    metadatas=[{"source": "fixed_control_test"}],
    ids=["chunk_1"]
)

# ==========================================
# 4. DEFINE THE AGENTS & TASKS (PIPELINE V2)
# ==========================================
scout = Agent(
    role='Data Scout',
    goal='Read the context and extract exactly 3 specific user complaints.',
    backstory='You are a strict data analyst. You only report facts. You must back up every claim with a direct quote from the text.',
    allow_delegation=False,
    llm=llm
)

architect = Agent(
    role='Solution Architect',
    goal='Create 1 specific product idea that solves the 3 complaints identified by the Scout.',
    backstory='You are a pragmatic product manager. Identify the specific target audience (the market gap) and propose a business model or product feature that solves their exact pain points.',
    allow_delegation=False,
    llm=llm
)

analyst = Agent(
    role='Competitor Analyst',
    goal='Search the web for companies that already do what the Architect proposed.',
    backstory='You are a ruthless market researcher. Use the search tool to find existing competitors.',
    tools=[search_tool],
    allow_delegation=False,
    llm=llm
)

critic = Agent(
    role='Final Evaluator',
    goal='Review the product idea and competitors. Give a GREEN LIGHT or RED LIGHT verdict.',
    backstory='You are a harsh venture capitalist. If competitors exist, RED LIGHT. If the idea is unique, GREEN LIGHT. Include the exact source quotes in your final report.',
    allow_delegation=False,
    llm=llm
)

task1 = Task(
    description=f"Analyze this data:\n{fixed_complaint_data}\n\nList 3 complaints. You MUST include a direct, word-for-word quote for each.",
    expected_output="A bulleted list of 3 complaints, each with an 'Evidence:' sub-bullet containing the exact quote.",
    agent=scout
)

task2 = Task(
    description="Propose 1 product idea to solve the 3 complaints. Explicitly define the target audience and how this fills a market gap.",
    expected_output="A paragraph describing the product, target audience, and how it fills the gap.",
    agent=architect
)

task3 = Task(
    description="Use your search tool to find at least 2 existing competitors for the proposed product.",
    expected_output="A list of existing competitors.",
    agent=analyst
)

task4 = Task(
    description="Write a final 'GREEN LIGHT' or 'RED LIGHT' report. You MUST include the direct quotes found by the Data Scout to prove the market gap is real.",
    expected_output="A final report with the verdict, competitor list, and a 'Source Evidence' section.",
    agent=critic
)

market_crew = Crew(
    agents=[scout, architect, analyst, critic],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential 
)

print("\n🚀 Kicking off the evaluation test...\n")
result = market_crew.kickoff()

print("\n==========================================")
print(f"🎯 FINAL OUTPUT ({engine_model})")
print("==========================================")
print(result)