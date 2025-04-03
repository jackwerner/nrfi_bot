import tweepy
import os
import time
from datetime import datetime

def get_acronym(team_name):
    # Define a mapping of team names to their abbreviations
    team_abbreviations = {
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",        
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Athletics": "ATH",
        "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD",
        "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH"
    }
    
    # Also add the abbreviations themselves as keys to handle cases where abbreviations are passed in
    abbreviation_map = {v: v for v in team_abbreviations.values()}
    team_abbreviations.update(abbreviation_map)
    
    return team_abbreviations.get(team_name, team_name)

def get_emoji(team_name):
    # Define a mapping of team names to their emojis
    team_emojis = {
        "Arizona Diamondbacks": "🐍",
        "Atlanta Braves": "🪶",        
        "Baltimore Orioles": "🐦‍⬛",
        "Boston Red Sox": "🔴🧦",
        "Chicago Cubs": "🐻",
        "Chicago White Sox": "⚪🧦",
        "Cincinnati Reds": "🔴",
        "Cleveland Guardians": "🛡️",
        "Colorado Rockies": "⛰️",
        "Detroit Tigers": "🐯",
        "Houston Astros": "⭐",
        "Kansas City Royals": "👑",
        "Los Angeles Angels": "👼",
        "Los Angeles Dodgers": "🔷",
        "Miami Marlins": "🐟",
        "Milwaukee Brewers": "🍺",
        "Minnesota Twins": "👬",
        "New York Mets": "🍎",
        "New York Yankees": "🗽",
        "Athletics": "🏅",
        "Philadelphia Phillies": "🔔",
        "Pittsburgh Pirates": "🏴‍☠️",
        "San Diego Padres": "⛪",
        "San Francisco Giants": "🌉",
        "Seattle Mariners": "⚓",
        "St. Louis Cardinals": "🐦‍🔥",
        "Tampa Bay Rays": "⛱️",
        "Texas Rangers": "🤠",
        "Toronto Blue Jays": "🐦",
        "Washington Nationals": "🇺🇸"
    }
    
    # Use the acronym to look up the emoji
    return team_emojis.get(team_name, "")

def tweet_nrfi_probabilities(games, probabilities, model_threshold=0.5):
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    client = tweepy.Client(
        consumer_key=consumer_key, 
        consumer_secret=consumer_secret,
        access_token=access_token, 
        access_token_secret=access_token_secret
    )

    tweets_sent = 0

    for i, game in enumerate(games):
        game_time_str = game['game_time'].strftime("%I:%M %p ET") if 'game_time' in game else "Time N/A"
        
        game_info = (f"🛌 NRFI Alert 🛌\n"
                     f"Game: {game['away_team']}{get_emoji(game['away_team'])} @ {game['home_team']}{get_emoji(game['home_team'])}\n"
                     f"🕒 Time: {game_time_str}\n"
                     f"⚔️ Pitchers: {game['away_pitcher']} vs {game['home_pitcher']}\n"
                     f"💤 NRFI Probability: {probabilities[i]:.2%}\n\n"
                     f"#{get_acronym(game['away_team'])}vs{get_acronym(game['home_team'])} #NRFI #NRFIAlert")
        
        while True:
            try:
                response = client.create_tweet(text=game_info)
                tweets_sent += 1
                print(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
                time.sleep(10)
                break

            except tweepy.errors.TooManyRequests as e:
                reset_time = int(e.response.headers['x-rate-limit-reset'])
                reset_datetime = datetime.fromtimestamp(reset_time)
                wait_time = (reset_datetime - datetime.now()).total_seconds() + 1
                print(f"limit:{e.response.headers['x-rate-limit-reset']}, remaining:{e.response.headers['x-rate-limit-remaining']}")
                print(f"Rate limit exceeded. Waiting until {reset_datetime} ({wait_time:.0f} seconds)")
                time.sleep(wait_time)

            except tweepy.errors.TweepyException as e:
                print(f"Error posting tweet: {e}")
                break

    print(f"Total tweets sent: {tweets_sent}")
    return tweets_sent 

def tweet_top_nrfi_poll(games, probabilities, num_games=4):
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    client = tweepy.Client(
        consumer_key=consumer_key, 
        consumer_secret=consumer_secret,
        access_token=access_token, 
        access_token_secret=access_token_secret
    )

    # Get top games
    top_games = games[:num_games]  # games should already be sorted by probability
    top_probs = probabilities[:num_games]

    # Create tweet text with game summaries
    tweet_text = "🎲 Which game will be a NRFI today? 🛌\n\n"
    poll_options = []
    
    for i, (game, prob) in enumerate(zip(top_games, top_probs), 1):
        summary = f"{i}. {game['away_pitcher']} ({get_acronym(game['away_team'])}{get_emoji(game['away_team'])}) @ {game['home_pitcher']} ({get_acronym(game['home_team'])}{get_emoji(game['home_team'])}) ({prob:.0%})\n"
        tweet_text += summary
        # Use team vs team format for poll options
        poll_options.append(f"{get_acronym(game['away_team'])}{get_emoji(game['away_team'])} @ {get_acronym(game['home_team'])}{get_emoji(game['home_team'])}")

    try:
        response = client.create_tweet(
            text=tweet_text,
            poll_duration_minutes=24*60,  # 24 hours
            poll_options=poll_options
        )
        print(f"Poll tweet posted successfully! Tweet ID: {response.data['id']}")
        return 1
    except tweepy.errors.TweepyException as e:
        print(f"Error posting poll tweet: {e}")
        return 0 