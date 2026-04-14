# Market Gap Identifier

This project is a Python tool that uses local AI models to automatically find and evaluate market gaps by analyzing user complaints.

## Project Structure

- **generator.py**: The first script to run. It takes a broad topic, searches YouTube and articles for user complaints, transcribes the videos, and extracts potential market gaps into a JSON file.
- **main.py**: The main execution script. It takes the generated gaps and uses a crew of AI agents to evaluate if the gap is truly unmet and valid. It saves the results to a CSV.
- **data_ingestion.py**: A helper file used by generator.py to handle searching YouTube and cleaning the transcript text.
- **metrics.py**: A testing script that compares the AI's evaluations against a human verified answer key to calculate accuracy, precision, and recall.
- **Testing_Gaps.csv**: The human verified answer key used by metrics.py.
- **generated_gaps.json** / **generated_gaps_sample.json**: The data files created by generator.py and read by main.py.

## Setup Instructions

1. Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```
2. Make sure you have Ollama installed and running locally on your machine with the necessary models pulled.

3. Create a .env file in the main folder and add your Tavily API key for the web search agent:
   ```
   TAVILY_API_KEY=your_key_here
   ```

## How to Use

**Step 1: Find the Gaps**

Run the generator script. It will prompt you for a seed topic in the terminal

```
python generator.py
```

**Step 2: Evaluate the Gaps**

Once the gaps are successfully saved to generated_gaps.json, run the main execution script to have the AI agents evaluate them.

```
python main.py
```
