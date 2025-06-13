import os
import random
import anthropic
import tweepy
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Get a random delay between 0 and 120 minutes (2 hours)
# delay_minutes = random.randint(0, 120)

# print(f"Waiting for {delay_minutes} minutes before posting...")

# Sleep for that random amount of time
# time.sleep(delay_minutes * 60)

def generate_human_tweet():
    """
    Use Claude API to generate a human-like tweet about baseball
    """
    # Initialize Claude client
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Read the last tweets if they exist
    last_tweets = []
    tweet_file = "./last_tweets.txt"
    if os.path.exists(tweet_file):
        with open(tweet_file, "r") as f:
            last_tweets = [line.strip() for line in f.readlines() if line.strip()]
    
        # Create the last tweets part of the prompt conditionally
    last_tweets_text = ""
    if last_tweets:
        last_tweets_text = "Here are the last few tweets we posted, make sure you write about something different:\n"
        for i, tweet in enumerate(last_tweets[-6:], 1):  # Get up to 6 most recent tweets
            # Split the tweet into words and take first 6
            words = tweet.split()
            truncated_tweet = ' '.join(words[:6]) + "..."
            last_tweets_text += f"{i}. \"{truncated_tweet}\"\n"
    
    prompt = f"""
    You are a neutral baseball observer writing tweets for male baseball fans in their 20s-30s. Write a short tweet (under 280 characters) about a current MLB story from yesterday or today. 

    Requirements: 
    - Focus on breaking news, standout performances, trades, injuries, or playoff implications 
    - Use a straightforward, conversational tone without exclamation points - No hashtags, internet slang, or "lol" type language 
    - Maintain complete neutrality 
    - never use "us," "our," or "we" when referring to teams 
    - Keep it direct and authentic, not overly animated or cute 
    - No emojis 
    - Do not cite sources or plagiarize content 
    - Stick to facts about the MLB news 
    - No philosophical observations or generalizations about baseball 

    Format: Just the tweet text, ready to post 

    Current date: {datetime.now().strftime("%Y-%m-%d")}.

    Examples of tone: 
    - "[Player] just went 4-for-4 with 2 home runs against [Team]. Guy's been locked in since the All-Star break" 
    - "[Team] traded [Player] to [Team] for [Player] and two prospects. That's a lot to give up for a rental" 
    - "[Player] placed on 15-day IL with shoulder inflammation. [Team] already thin at that position" 
    IMPORTANT: Do not include any introduction or explanation. Write ONLY the tweet and nothing else.

    {last_tweets_text}
    
    IMPORTANT: Do not include any introduction or explanation. Write ONLY the tweet and nothing else.

    Write the tweet:
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=120,
            temperature=0.5,
            system="""
                You are a baseball fan in your 20s-30s with reasonable opinions. 
                Write in a casual, conversational tone as if texting a friend. 
                Be moderately opinionated but natural. 
                Don't use hashtags, 'lol', internet slang, or exclamation points. 
                Don't refer to any team as 'we', 'us', or 'our'. 
                Keep it extremely concise - 1-2 punchy sentences maximum. 
                Tone: conversational, concise, simplify the language. Act as if you're speaking to a close friend.
                Your task is to write ONLY the tweet text with no introduction or explanation.
                """,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    # Optional: Limit searches per request
                    "max_uses": 3,
                    # Optional: Only include results from these domains
                    "allowed_domains": ["mlb.com", "mlbtraderumors.com","ftnfantasy.com","pitcherlist.com","yardbarker.com","espn.com","apnews.com","cbssports.com","yahoo.com","sports.yahoo.com","si.com"],
                }
            ]
        )
        
        # Extract the generated tweet from the response
        tweet_text = response.content[-1].text.strip()
        
        # Backup parsing logic to handle cases with introductory text
        if "Based on" in tweet_text or "I'll create" in tweet_text or "Looking at" in tweet_text:
            # Look for the actual tweet after introductory text
            lines = tweet_text.split('\n')
            for line in lines:
                # Skip empty lines and lines that look like introductions
                if not line.strip() or "Based on" in line or "I'll create" in line or "Looking at" in line:
                    continue
                # The first non-empty, non-introduction line is likely the tweet
                tweet_text = line.strip()
                break
        
        # Save the generated tweet to the file with the last 4 tweets
        if last_tweets:
            # Keep only the 3 most recent tweets and add the new one
            last_tweets = last_tweets[-5:] + [tweet_text]
        else:
            last_tweets = [tweet_text]
            
        with open(tweet_file, "w") as f:
            f.write("\n".join(last_tweets))
        
        return tweet_text
    
    except Exception as e:
        print(f"Error generating tweet with Claude: {e}")
        return None

def post_random_human_tweet():
    """
    Generate and post a human-like tweet about baseball
    """
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        raise ValueError("Twitter API credentials not set in environment variables")

    client = tweepy.Client(
        consumer_key=consumer_key, 
        consumer_secret=consumer_secret,
        access_token=access_token, 
        access_token_secret=access_token_secret
    )

    # Generate the tweet
    tweet_text = generate_human_tweet()
    if not tweet_text:
        print("Failed to generate tweet")
        return False

    print(tweet_text)
    # Post the tweet
    try:
        response = client.create_tweet(text=tweet_text)
        print(f"Human-like tweet posted successfully! Tweet ID: {response.data['id']}")
        print(f"Tweet content: {tweet_text}")
        return True
    except tweepy.errors.TooManyRequests as e:
        reset_time = int(e.response.headers['x-rate-limit-reset'])
        reset_datetime = datetime.fromtimestamp(reset_time)
        wait_time = (reset_datetime - datetime.now()).total_seconds() + 1
        print(f"Rate limit exceeded. Waiting until {reset_datetime} ({wait_time:.0f} seconds)")
        time.sleep(wait_time)
        return False
    except tweepy.errors.TweepyException as e:
        print(f"Error posting tweet: {e}")
        return False

# if __name__ == "__main__":
#     post_random_human_tweet()
post_random_human_tweet()
