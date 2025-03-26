import os
import dotenv
from datetime import datetime, timedelta
import pandas as pd

# Import modules from our package
from nrfi_bot.data.mlb_api import get_todays_games, get_team_stats, get_pitcher_stats, get_season_games
from nrfi_bot.data.data_processor import prepare_data, format_game_info
from nrfi_bot.models.nrfi_model import train_model, predict_nrfi_probabilities
from nrfi_bot.utils.twitter import tweet_nrfi_probabilities

def main():
    # Load environment variables
    dotenv.load_dotenv()
    
    # current_year = datetime.now().year
    # season_start = datetime(current_year, 4, 1)
    # yesterday = datetime.now() - timedelta(days=1)
    season_start = datetime(2024, 4, 1)
    current_year = 2024
    yesterday = datetime(2024, 9, 30)

    # Get historical data for model training
    season_games = get_season_games(season_start, yesterday)
    team_stats = get_team_stats()
    pitcher_stats = get_pitcher_stats()
    historical_data = prepare_data(season_games, team_stats, pitcher_stats)
    
    # Train the model
    model = train_model(historical_data)
    
    # Get today's games and prepare data for prediction
    today_games = get_todays_games()
    today_data = prepare_data(today_games, team_stats, pitcher_stats)
    
    # Make predictions
    probabilities = predict_nrfi_probabilities(model, today_data)
    
    # Match predictions with game data
    game_prob_pairs = []
    for i, original_game in enumerate(today_games):
        matching_row = today_data[(today_data['home_team'] == original_game['home_team']) & 
                                  (today_data['away_team'] == original_game['away_team'])]
        
        if not matching_row.empty:
            game_info = matching_row.iloc[0].to_dict()
            game_info['game_time'] = original_game['game_time']
            
            row_index = matching_row.index[0]
            probability = probabilities[row_index]
            
            game_prob_pairs.append((game_info, probability))
        else:
            print(f"\nWarning: No matching prepared data for game {i + 1}: {original_game['away_team']} @ {original_game['home_team']}")
    
    # Sort games by NRFI probability
    sorted_games = sorted(game_prob_pairs, key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nAll Games Sorted by NRFI Probability:")
    for game, prob in sorted_games:
        threshold_info = f"(Model threshold: {model['optimal_threshold']:.2f})"
        prediction_text = "PREDICTED NRFI" if prob >= model['optimal_threshold'] else "PREDICTED YRFI"
        print(f"Probability: {prob} {threshold_info}")
        print(format_game_info(game, prob))
        print(f"{prediction_text}")
        print()

    # Uncomment to enable tweeting
    tweets_sent = tweet_nrfi_probabilities([game for game, _ in sorted_games], 
                                           [prob for _, prob in sorted_games],
                                           model['optimal_threshold'])
    print(f"Total tweets sent in this run: {tweets_sent}")

if __name__ == "__main__":
    main()
