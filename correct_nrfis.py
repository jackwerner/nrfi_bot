import os
from datetime import datetime, timedelta
import pandas as pd
import tweepy
from dotenv import load_dotenv
from analyze_predictions import analyze_predictions, detailed_analysis
from utils.twitter import get_acronym, get_emoji
import random

def tweet_correct_predictions():
    """
    Analyze yesterday's predictions and tweet about the correct ones
    """
    # Load environment variables
    load_dotenv()
    
    # Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    
    # Get detailed analysis for yesterday
    analysis = detailed_analysis(yesterday, yesterday)
    
    if not analysis or 'detailed_results' not in analysis or analysis['detailed_results'].empty:
        print(f"No prediction data found for {yesterday.strftime('%Y-%m-%d')}")
        return
    
    # Filter for correct predictions only
    correct_predictions = analysis['detailed_results'][analysis['detailed_results']['correct'] == True]
    
    if correct_predictions.empty:
        print("No correct predictions found for yesterday")
        return
    
    # Set up Twitter client
    client = tweepy.Client(
        consumer_key=os.getenv("CONSUMER_KEY"),
        consumer_secret=os.getenv("CONSUMER_SECRET"),
        access_token=os.getenv("ACCESS_TOKEN"),
        access_token_secret=os.getenv("ACCESS_TOKEN_SECRET")
    )
    
    # First, tweet a summary
    total_predictions = len(analysis['detailed_results'])
    correct_count = len(correct_predictions)
    accuracy = correct_count / total_predictions
    
    # Spice up the summary tweet based on accuracy
    if accuracy >= 0.7:
        summary_emoji = "🔥🎯"
        summary_descriptor = "CRUSHING IT"
    elif accuracy >= 0.5:
        summary_emoji = "💪⚾"
        summary_descriptor = "SOLID DAY"
    else:
        summary_emoji = "📈⚾"
        summary_descriptor = "BUILDING MOMENTUM"
    
    summary_tweet = (
        f"{summary_emoji} {summary_descriptor} WITH NRFI PREDICTIONS! {summary_emoji}\n"
        f"🎯 Nailed {correct_count}/{total_predictions} predictions ({accuracy:.1%})\n"
        f"📅 {yesterday.strftime('%B %d, %Y')}\n"
        f"🔥 #NRFI  #NRFIAlert #MLB #BaseballBetting"
    )
    
    try:
        summary_response = client.create_tweet(text=summary_tweet)
        print(f"Summary tweet posted successfully! Tweet ID: {summary_response.data['id']}")
        
        # Wait a bit before posting individual game results
        import time
        time.sleep(5)
        
        # Now tweet about each correct prediction
        for _, prediction in correct_predictions.iterrows():
            away_team = get_acronym(prediction['away_team'])
            home_team = get_acronym(prediction['home_team'])
            away_emoji = get_emoji(prediction['away_team'])
            home_emoji = get_emoji(prediction['home_team'])
            
            # Add variety to the correct prediction tweets
            celebration_phrases = [
                "🎯 BULLSEYE! Nailed the",
                "🔥 MONEY! Called the", 
                "💰 CASH! Predicted the",
                "⚡ BOOM! Hit the",
                "🎪 MAGIC! Conjured the",
                "✨ PERFECT! Crushed the",
                "🎲 JACKPOT! Landed the",
                "🎪 SHOWTIME! Delivered the",
                "🌟 STELLAR! Locked in the",
                "💫 CLUTCH! Secured the",
                "🎯 PRECISION! Predicted the",
            ]
            
            confidence_descriptions = [
                "with ice-cold confidence",
                "like a seasoned pro",
                "with surgical precision",
                "with unwavering conviction", 
                "with crystal clear vision",
                "with laser focus",
                "with pinpoint accuracy",
                "like a hawk spotting prey",
                "like clockwork",
                "with machine-like precision",
                "like a chess grandmaster",
                "with sniper-like accuracy",
                "like a fortune teller",
                "with supercomputer certainty",
            ]
            
            celebration = random.choice(celebration_phrases)
            confidence_desc = random.choice(confidence_descriptions)
            
            result_tweet = (
                f"{celebration} {prediction['predicted']}! Another successful prediction yesterday...\n"
                f"🏟️ {away_team}{away_emoji} @ {home_team}{home_emoji}\n"
                f"⚾ Hurlers: {prediction['away_pitcher']} vs {prediction['home_pitcher']}\n"
                f"🎯 Called it {confidence_desc} ({prediction['nrfi_probability']:.1%})\n"
                f"#{get_acronym(prediction['away_team'])}vs{get_acronym(prediction['home_team'])} #NRFI #NRFIAlert #BaseballBetting"
            )
            
            try:
                response = client.create_tweet(text=result_tweet)
                print(f"Result tweet posted successfully! Tweet ID: {response.data['id']}")
                time.sleep(10)  # Wait between tweets to avoid rate limits
            except tweepy.errors.TweepyException as e:
                print(f"Error posting result tweet: {e}")
                
    except tweepy.errors.TweepyException as e:
        print(f"Error posting summary tweet: {e}")

if __name__ == "__main__":
    tweet_correct_predictions()
