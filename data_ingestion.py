import spacy
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW: Built-in Python libraries (no pip install required!)
import urllib.request
import urllib.parse
import re

# 1. Load the NLP model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

def clean_text(raw_text):
    """Removes junk and stop words."""
    doc = nlp(raw_text)
    clean_tokens = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct 
                    and not token.like_url and token.is_alpha]
    return " ".join(clean_tokens)

def chunk_data(text):
    """Chops massive text blocks into smaller, overlapping paragraphs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def get_youtube_video_ids(query, limit=3):
    """Searches YouTube using raw HTML and Regex to bypass dependency conflicts."""
    print(f"Searching YouTube for: '{query}'...")
    try:
        # Format the search query for a URL
        encoded_query = urllib.parse.quote(query)
        html = urllib.request.urlopen(f"https://www.youtube.com/results?search_query={encoded_query}")
        
        # Use Regex to find the 11-character Video IDs in the raw HTML
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        
        # Remove duplicates while keeping the order of top results
        unique_ids = list(dict.fromkeys(video_ids))
        return unique_ids[:limit]
    except Exception as e:
        print(f"YouTube search failed: {e}")
        return []

def fetch_youtube_transcripts(video_ids):
    """Fetches, combines, and cleans transcripts from multiple videos."""
    ytt_api = YouTubeTranscriptApi()
    all_text = ""
    
    for vid in video_ids:
        print(f"Fetching transcript for video: {vid}...")
        try:
            transcript = ytt_api.fetch(vid).to_raw_data()
            full_text = " ".join([entry['text'] for entry in transcript])
            all_text += full_text + " "
        except Exception as e:
            print(f"Skipping {vid} (No transcript available or subtitles disabled)")
            
    if not all_text.strip():
        return [] 
        
    cleaned = clean_text(all_text)
    return chunk_data(cleaned)

# --- Testing the Harvester ---
if __name__ == "__main__":
    test_query = "user complaints about meal prep delivery services"
    
    # 1. Find the videos
    vids = get_youtube_video_ids(test_query, limit=2)
    print(f"Found Videos: {vids}")
    
    # 2. Get the transcripts and chunk them
    chunks = fetch_youtube_transcripts(vids)
    
    print(f"\nSuccessfully created {len(chunks)} text chunks!")
    if chunks:
        print("\n--- Snippet of Chunk 1 ---")
        print(chunks[0][:300] + "...")