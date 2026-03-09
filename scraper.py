import requests
import pandas as pd
import datetime

# --- UNIVERSAL HEADER (To trick Reddit into thinking we are a browser) ---
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def fetch_reddit_search(topic, limit=20):
    """Searches Reddit for a generic topic (e.g., 'iPhone 16')"""
    print(f"📡 Searching Reddit (JSON Mode) for: '{topic}'...")
    url = f"https://www.reddit.com/search.json?q={topic}&sort=relevance&limit={limit}"
    
    return _process_response(url, source_type="search")

def fetch_reddit_comments(post_url):
    """Fetches comments from a SPECIFIC Reddit Post URL"""
    print(f"🔗 Connecting to specific post: {post_url}...")
    
    # Ensure the URL ends with .json
    clean_url = post_url.split('?')[0] # Remove query parameters
    if not clean_url.endswith('.json'):
        target_url = clean_url + ".json"
    else:
        target_url = clean_url

    return _process_response(target_url, source_type="post")

def _process_response(url, source_type):
    """Helper function to handle the JSON logic"""
    try:
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"❌ Error: Reddit blocked us or URL is wrong. Status: {response.status_code}")
            return None

        data = response.json()
        parsed_data = []

        if source_type == "search":
            # Handle Search Results
            posts = data['data']['children']
            for post in posts:
                p = post['data']
                parsed_data.append({
                    "title": p.get('title', ''),
                    "body": p.get('selftext', 'Check link'),
                    "score": p.get('score', 0),
                    "num_comments": p.get('num_comments', 0),
                    "url": p.get('url', '')
                })

        elif source_type == "post":
            # Handle Specific Post (Index 0 is post, Index 1 is comments)
            # 1. Get the Main Post Title
            main_post = data[0]['data']['children'][0]['data']
            post_title = main_post.get('title', 'Unknown Title')
            
            # 2. Get the Comments
            comments = data[1]['data']['children']
            
            for comment in comments:
                c = comment.get('data', {})
                # Skip "more" tags (which aren't real comments)
                if 'body' in c:
                    parsed_data.append({
                        "title": post_title, # Repeat title so analyzer knows the context
                        "body": c.get('body', ''), # The actual comment
                        "score": c.get('score', 0),
                        "num_comments": 0, # Comments don't usually have sub-comments count easily accessible here
                        "url": main_post.get('url', '')
                    })

        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)
        print(f"✅ Success! Extracted {len(parsed_data)} items.")
        return df

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return None