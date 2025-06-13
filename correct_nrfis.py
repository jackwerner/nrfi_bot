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
        f"{summary_emoji} {summary_descriptor} WITH THE MODEL! {summary_emoji}\n"
        f"🎯 Nailed {correct_count}/{total_predictions} predictions ({accuracy:.1%})\n"
        f"📅 {yesterday.strftime('%B %d, %Y')}\n"
        f"🔥 #NRFI #MLB"
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
            
            # Special celebration phrases for NRFI predictions
            nrfi_celebration_phrases = [
                "😴 SNOOZE FEST! Called the",
                "🥱 BORING! Nailed the",
                "💤 SNOOZERS! Predicted the",
                "🛏️ SLEEPY TIME! Got the",
                "😪 YAWN! Secured the",
                "🌙 NAPTIME! Delivered the",
                "🧘‍♂️ ZEN MODE! Called the",
                "📺 CHANNEL CHANGER! Predicted the",
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
            
            # Choose celebration phrase based on prediction type
            if prediction['predicted'] == 'NRFI':
                celebration = random.choice(nrfi_celebration_phrases)
            else:
                celebration = random.choice(celebration_phrases)
            
            confidence_desc = random.choice(confidence_descriptions)
            
            # Get first inning stats from the game data
            from data.mlb_api import get_season_games
            game_data = get_season_games(
                datetime.strptime(prediction['date'], '%Y-%m-%d'),
                datetime.strptime(prediction['date'], '%Y-%m-%d')
            )
            
            # Find the matching game
            matching_game = None
            for game in game_data:
                if (game['home_team'] == prediction['home_team'] and 
                    game['away_team'] == prediction['away_team'] and
                    game['home_pitcher'] == prediction['home_pitcher'] and
                    game['away_pitcher'] == prediction['away_pitcher']):
                    matching_game = game
                    break
            
            if not matching_game:
                print("No matching game found!")

            # Get first inning stats if available
            first_inning_stats = ""
            if matching_game and 'linescore' in matching_game and matching_game['linescore'] and 'innings' in matching_game['linescore'] and matching_game['linescore']['innings']:
                first_inning = matching_game['linescore']['innings'][0]
                away_hits = first_inning['away'].get('hits', 0)
                away_lob = first_inning['away'].get('leftOnBase', 0)
                away_errors = first_inning['away'].get('errors', 0)
                home_hits = first_inning['home'].get('hits', 0)
                home_lob = first_inning['home'].get('leftOnBase', 0)
                home_errors = first_inning['home'].get('errors', 0)
                total_hits = away_hits + home_hits
                total_lob = away_lob + home_lob
                total_errors = away_errors + home_errors
                
                # Add drama level based on hits and LOB
                drama_level = "😅 Close Call!" if (total_hits > 1 or total_errors > 0 or total_lob > 0) else "😎 Easy Money!"
                
                first_inning_stats = (
                    f"\n📊 1st Inning Stats:\n"
                    f"💥 Hits: {away_hits} ({away_team}) | {home_hits} ({home_team})\n"
                    f"❌ Errors: {away_errors} ({away_team}) | {home_errors} ({home_team})\n"
                    f"🏃 LOB: {away_lob} ({away_team}) | {home_lob} ({home_team})\n"
                    f"{drama_level}"
                )
            else:
                print("No linescore data available for this game")
            
            result_tweet = (
                f"{celebration} {prediction['predicted']}!\n"
                f"🏟️ {away_team}{away_emoji} @ {home_team}{home_emoji}\n"
                f"⚾ {prediction['away_pitcher']} vs {prediction['home_pitcher']}\n"
                f"🎯 Predicted likelihood: {prediction['nrfi_probability']:.1%}"
                f"\n{first_inning_stats}\n"
                f"#{get_acronym(prediction['away_team'])}vs{get_acronym(prediction['home_team'])}"
            )
            
            try:
                response = client.create_tweet(text=result_tweet)
                print(f"Result tweet posted successfully! Tweet ID: {response.data['id']}")
                print(result_tweet)
                time.sleep(5)  # Wait between tweets to avoid rate limits
            except tweepy.errors.TweepyException as e:
                print(f"Error posting result tweet: {e}")
                
    except tweepy.errors.TweepyException as e:
        print(f"Error posting summary tweet: {e}")

if __name__ == "__main__":
    tweet_correct_predictions()
