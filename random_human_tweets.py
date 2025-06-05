import os
import random
import anthropic
import tweepy
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Get a random delay between 0 and 120 minutes (2 hours)
delay_minutes = random.randint(0, 120)

print(f"Waiting for {delay_minutes} minutes before posting...")

# Sleep for that random amount of time
time.sleep(delay_minutes * 60)

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
        last_tweets_text = "Here are the last few tweets I posted, make sure you tweet about something different:\n"
        for i, tweet in enumerate(last_tweets[-6:], 1):  # Get up to 6 most recent tweets
            last_tweets_text += f"{i}. \"{tweet}\"\n"
    
    prompt = f"""
    Today's date is {datetime.now().strftime("%Y-%m-%d")}.
    Write a short tweet as a baseball fan observing or sharing thoughts on a current MLB story. 
    Focus on an immediately relevant topic from yesterday or today.

    Sound like a real, natural person with genuine opinions - not a corporate account or news headline. 
    Be conversational and personal in your tone. Use more casual language and first-person perspective.
    Be moderately opinionated and insightful but keep it observational and conversational.
    Your target audience is men in their 20s-30s who are baseball fans. Don't be super animated, don't try to be cute. 
    
    You do not have a favorite team, you are a neutral observer. Don't refer to any team as 'us', 'our', 'we'.
    Focus on ONE specific topic in 1-2 short, punchy sentences. The less words you use, the better.
    
    Avoid sounding like a news headline or report. You do not need to summarize the topic. 
    No "lol" or internet slang. No exclamation points. Do not use hashtags. Do not plagiarize. Do not cite your sources.
    
    IMPORTANT: Keep it extremely concise and brief. Avoid adding philosophical or reflective statements at the end.
    
    VARIETY GUIDELINES:
    - Mix up your opening structure - don't always start with "Player Name + action"
    - Vary between player-focused, team-focused, and situation-focused observations
    - Avoid cliché phrases like "peak [team] baseball", "flair for the dramatic", "hits different"
    - Try different angles: injury reactions, strategic observations, season narrative, individual performances
    
    Good Examples:
    - 'Still thinking about Taveras crushing that two-run shot in the 8th to put Seattle ahead yesterday. Mariners quietly building momentum with five wins in their last six games while everyone's focused elsewhere.'
    - 'Orioles finally snap that brutal eight-game skid with an extra innings win against the Brewers. Mansolino gets his first W as interim manager but definitely had to sweat through some late blown leads to get there.'
    - 'Blue Jays absolutely dismantled the Padres yesterday with that 14-0 shutout. When Toronto's offense gets rolling like that, makes you wonder why they've been so inconsistent all season.'
    - 'That Kirby injury last night was hard to watch. 102 mph right to the face but seeing him walk off on his own was at least somewhat reassuring.'
    - 'Rockies swept Miami for their first series win in two months. Sometimes the worst teams still find ways to surprise you.'
    - 'Dodgers keep finding ways to win these close games. Freeman's walk-off double was clutch but they really shouldn't have needed extra innings against this Mets team.'
 
    {last_tweets_text}
    
    IMPORTANT: Do not include any introduction or explanation. Write ONLY the tweet and nothing else.
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
                    "max_uses": 1,
                    # Optional: Only include results from these domains
                    "allowed_domains": ["mlb.com", "mlbtraderumors.com","ftnfantasy.com","pitcherlist.com","yardbarker.com","espn.com","apnews.com","cbssports.com"],
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
