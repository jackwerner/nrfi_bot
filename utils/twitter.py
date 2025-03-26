import tweepy
import os
import time
from datetime import datetime

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
        prediction = "NRFI" if probabilities[i] >= model_threshold else "YRFI"
        
        game_info = (f"⚾ NRFI Probability ⚾\n"
                     f"{game['away_team']} @ {game['home_team']} - {game_time_str}⏰\n"
                     f"Pitchers: {game['away_pitcher']} 🆚 {game['home_pitcher']}\n"
                     f"NRFI Probability: {probabilities[i]:.2%} {'📈' if probabilities[i] > 0.5 else '📉' if probabilities[i] < 0.5 else '⚖️'}\n"
                     f"Prediction: {prediction} (Threshold: {model_threshold:.2f})")
        
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