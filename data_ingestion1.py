import praw
from youtube_transcript_api import YouTubeTranscriptApi
import spacy

# 1. Load the NLP model for text cleaning
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

def clean_text(raw_text):
    """
    Passes raw text through spaCy.
    Removes URLs, punctuation, and 'stop words' (the, is, at, which, etc.).
    Returns a dense, lowercase string of useful keywords.
    """
    doc = nlp(raw_text)
    clean_tokens = []
    
    for token in doc:
        # Filter out junk: URLs, punctuation, stop words, and numbers
        if not token.is_stop and not token.is_punct and not token.like_url and token.is_alpha:
            # token.lemma_ gets the root word (e.g., 'running' becomes 'run')
            clean_tokens.append(token.lemma_.lower())
            
    return " ".join(clean_tokens)

def fetch_youtube_data(video_id):
    """Fetches and cleans a YouTube video transcript."""
    print(f"Fetching transcript for YouTube video: {video_id}...")
    try:
        # Initialize the API client (New syntax for v1.2+)
        ytt_api = YouTubeTranscriptApi()
        
        # Fetch the transcript and convert to the raw dictionary format
        transcript = ytt_api.fetch(video_id).to_raw_data()
        
        # Combine all the timestamped text into one giant string
        full_text = " ".join([entry['text'] for entry in transcript])
        return clean_text(full_text)
    except Exception as e:
        return f"Error fetching YouTube: {e}"
    

def fetch_reddit_data(subreddit_name, search_query, limit=5):
    """Fetches and cleans Reddit comments from a specific search query."""
    print(f"Fetching Reddit data from r/{subreddit_name} for query: '{search_query}'...")
    
    # ADD YOUR CREDENTIALS HERE
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="mac:market-gap-identifier:v1.0 (by /u/YOUR_REDDIT_USERNAME)"
    )
    
    raw_comments = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Search the subreddit for relevant posts
        for submission in subreddit.search(search_query, limit=limit):
            submission.comments.replace_more(limit=0) # Flatten comment trees
            for comment in submission.comments.list():
                raw_comments.append(comment.body)
                
        full_reddit_text = " ".join(raw_comments)
        return clean_text(full_reddit_text)
    except Exception as e:
        return f"Error fetching Reddit: {e}"

# --- Testing the Pipeline ---
if __name__ == "__main__":
    # Test 1: YouTube (Using a random Marques Brownlee tech review video ID)
    # The ID is the part of the URL after "v=" (e.g., youtube.com/watch?v=dQw4w9WgXcQ)
    yt_video_id = "eFUB_jL_XcM" 
    cleaned_yt = fetch_youtube_data(yt_video_id)
    
    print("\n--- Cleaned YouTube Output Snippet ---")
    print(cleaned_yt[:500] + "...\n") 
    
    # Test 2: Reddit (You must add your API keys above for this to work)
    # cleaned_reddit = fetch_reddit_data(subreddit_name="startups", search_query="crm software complaints")
    # print("\n--- Cleaned Reddit Output Snippet ---")
    # print(cleaned_reddit[:500] + "...\n")