import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
import tweepy
import numpy as np
import json

def get_todays_games():
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hydrate": "team,probablePitcher"
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        games = []
        for date in data['dates']:
            for game in date['games']:
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName')
                
                if home_pitcher and away_pitcher and home_pitcher != 'TBD' and away_pitcher != 'TBD':
                    games.append({
                        'game_id': game['gamePk'],
                        'home_team': game['teams']['home']['team']['name'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_team_id': game['teams']['home']['team']['id'],
                        'away_team_id': game['teams']['away']['team']['id'],
                        'home_pitcher': home_pitcher,
                        'away_pitcher': away_pitcher
                    })
        return games
    else:
        raise Exception(f"Failed to fetch games: {response.status_code}")

def get_team_stats():
    url = "https://statsapi.mlb.com/api/v1/teams"
    params = {
        "sportId": 1,
        "season": datetime.now().year
    }
    response = requests.get(url, params=params)
    print(f"Team list API call: Status code {response.status_code}")
    if response.status_code == 200:
        teams = response.json()['teams']
        team_stats = {}
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            stats_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season={datetime.now().year}"
            stats_response = requests.get(stats_url)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()['stats'][0]['splits'][0]['stat']
                
                runs = stats_data.get('runs')
                games = stats_data.get('gamesPlayed')
                
                if runs is not None and games is not None and games > 0:
                    runs_scored_per_game = runs / games
                else:
                    runs_scored_per_game = None
                    print(f"Unable to calculate runsScoredPerGame for team {team_name} (ID: {team_id})")
                
                team_stats[team_id] = {
                    'name': team_name,
                    'runsScoredPerGame': runs_scored_per_game,
                    'ops': stats_data.get('ops')
                }
                
                for stat, value in team_stats[team_id].items():
                    if value is None:
                        print(f"Missing {stat} for team {team_name}")
            else:
                print(f"Failed to fetch stats for {team_name}: Status code {stats_response.status_code}")
    return team_stats

def get_pitcher_stats():
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season",
        "group": "pitching",
        "season": datetime.now().year,
        "playerPool": "All",  # Changed from "Qualified" to "All"
        "limit": 100,  # Increased limit
        "offset": 0
    }
    pitcher_stats = {}
    
    while True:
        response = requests.get(url, params=params)
        print(f"Pitcher stats API call: Status code {response.status_code}, Offset: {params['offset']}")
        
        if response.status_code == 200:
            data = response.json()
            for split in data['stats'][0]['splits']:
                pitcher_name = split['player']['fullName']
                pitcher_stats[pitcher_name] = {
                    'era': split['stat'].get('era'),
                    'whip': split['stat'].get('whip'),
                    'strikeoutsPer9Inn': split['stat'].get('strikeoutsPer9Inn'),
                    'ops': split['stat'].get('ops')
                }
                for stat, value in pitcher_stats[pitcher_name].items():
                    if value is None:
                        print(f"Missing {stat} for pitcher {pitcher_name}")
            
            # Check if we've retrieved all pitchers
            if len(data['stats'][0]['splits']) < params['limit']:
                break
            
            # Update offset for next page
            params['offset'] += params['limit']
        else:
            raise Exception(f"Failed to fetch pitcher stats: {response.status_code}")
    
    print(f"Total pitchers retrieved: {len(pitcher_stats)}")
    return pitcher_stats

def get_season_games(start_date, end_date):
    games = []
    current_date = start_date
    print(f"Getting season games from {start_date} to {end_date}")
    while current_date <= end_date:
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": current_date.strftime("%Y-%m-%d"),
            "hydrate": "team,probablePitcher,linescore"
        }
        response = requests.get(url, params=params)
        
        # print(f"Fetching games for {current_date.strftime('%Y-%m-%d')}: Status code {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            for date in data['dates']:
                for game in date['games']:
                    if 'linescore' in game and game['linescore']['innings']:
                        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName')
                        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName')
                        
                        if home_pitcher and away_pitcher and home_pitcher != 'TBD' and away_pitcher != 'TBD':
                            nrfi = 1 if game['linescore']['innings'][0]['away']['runs'] == 0 and \
                                        game['linescore']['innings'][0]['home']['runs'] == 0 else 0
                            games.append({
                                'game_id': game['gamePk'],
                                'home_team': game['teams']['home']['team']['name'],
                                'away_team': game['teams']['away']['team']['name'],
                                'home_team_id': game['teams']['home']['team']['id'],
                                'away_team_id': game['teams']['away']['team']['id'],
                                'home_pitcher': home_pitcher,
                                'away_pitcher': away_pitcher,
                                'nrfi': nrfi
                            })
                        else:
                            print(f"Skipped game due to missing pitcher info: {game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}")
                    else:
                        print(f"Skipped game due to missing linescore: {game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}")
        else:
            print(f"Failed to fetch games for {current_date.strftime('%Y-%m-%d')}: {response.status_code}")
        current_date += timedelta(days=1)
    
    print(f"Fetched {len(games)} games with known pitchers")
    return games

def prepare_data(games, team_stats, pitcher_stats):
    data = []
    excluded_games = 0
    missing_pitchers = set()
    
    for game in games:
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        home_pitcher = game['home_pitcher']
        away_pitcher = game['away_pitcher']
        
        # Check if both pitchers are in pitcher_stats
        if home_pitcher not in pitcher_stats:
            missing_pitchers.add(home_pitcher)
            excluded_games += 1
            continue  # Skip this game
        if away_pitcher not in pitcher_stats:
            missing_pitchers.add(away_pitcher)
            excluded_games += 1
            continue  # Skip this game
        
        if home_team_id in team_stats and away_team_id in team_stats:
            home_team_data = team_stats[home_team_id]
            away_team_data = team_stats[away_team_id]
            
            game_data = {
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_pitcher': home_pitcher,
                'away_pitcher': away_pitcher,
                'home_runsScoredPerGame': home_team_data['runsScoredPerGame'],
                'home_ops': home_team_data['ops'],
                'away_runsScoredPerGame': away_team_data['runsScoredPerGame'],
                'away_ops': away_team_data['ops'],
                'home_pitcher_era': pitcher_stats[home_pitcher]['era'],
                'home_pitcher_whip': pitcher_stats[home_pitcher]['whip'],
                'home_pitcher_strikeoutsPer9Inn': pitcher_stats[home_pitcher]['strikeoutsPer9Inn'],
                'home_pitcher_ops': pitcher_stats[home_pitcher]['ops'],
                'away_pitcher_era': pitcher_stats[away_pitcher]['era'],
                'away_pitcher_whip': pitcher_stats[away_pitcher]['whip'],
                'away_pitcher_strikeoutsPer9Inn': pitcher_stats[away_pitcher]['strikeoutsPer9Inn'],
                'away_pitcher_ops': pitcher_stats[away_pitcher]['ops']
            }
            
            # Only add the game if all stats are available
            if all(value is not None for value in game_data.values()):
                if 'nrfi' in game:
                    game_data['nrfi'] = game['nrfi']
                data.append(game_data)
            else:
                excluded_games += 1
        else:
            excluded_games += 1
    
    print(f"Excluded {excluded_games} games due to missing data")
    print(f"Missing stats for {len(missing_pitchers)} pitchers:")
    for pitcher in sorted(missing_pitchers):
        print(f"  {pitcher}")
    
    df = pd.DataFrame(data)
    
    # Convert all columns to float except for team names and pitcher names
    for col in df.columns:
        if col not in ['home_team', 'away_team', 'home_pitcher', 'away_pitcher']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    return df

def train_model(data):
    # Remove categorical columns
    X = data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1)
    y = data['nrfi']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with StandardScaler and LogisticRegression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(random_state=42))
    ])

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred_proba)
    print(f"R-squared score: {r2:.2f}")

    # Get feature importances (coefficients for logistic regression)
    feature_importance = model.named_steps['logistic'].coef_[0]
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")

    class_balance = y.value_counts(normalize=True)
    print("Class balance:")
    print(class_balance)

    return model

def predict_nrfi_probabilities(model, today_data):
    features = today_data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1, errors='ignore')
    probabilities = model.predict_proba(features)[:, 1]  # Probability of NRFI (class 1)
    return dict(enumerate(probabilities))

def format_game_info(game, probability):
    
    return (f"{game['away_team']} @ {game['home_team']}\n"
            f"Pitchers: {game['away_pitcher']} vs {game['home_pitcher']}\n"
            f"NRFI Probability: {probability:.2%}\n"
            f"Home Team Stats:\n"
            f"  Runs Scored Per Game: {game.get('home_runsScoredPerGame', 'N/A'):.4f}\n"
            f"  OPS: {game.get('home_ops', 'N/A'):.3f}\n"
            f"Away Team Stats:\n"
            f"  Runs Scored Per Game: {game.get('away_runsScoredPerGame', 'N/A'):.4f}\n"
            f"  OPS: {game.get('away_ops', 'N/A'):.3f}\n"
            f"Home Pitcher Stats:\n"
            f"  ERA: {game.get('home_pitcher_era', 'N/A'):.2f}\n"
            f"  WHIP: {game.get('home_pitcher_whip', 'N/A'):.2f}\n"
            f"  K/9: {game.get('home_pitcher_strikeoutsPer9Inn', 'N/A'):.2f}\n"
            f"  OPS Against: {game.get('home_pitcher_ops', 'N/A'):.3f}\n"
            f"Away Pitcher Stats:\n"
            f"  ERA: {game.get('away_pitcher_era', 'N/A'):.2f}\n"
            f"  WHIP: {game.get('away_pitcher_whip', 'N/A'):.2f}\n"
            f"  K/9: {game.get('away_pitcher_strikeoutsPer9Inn', 'N/A'):.2f}\n"
            f"  OPS Against: {game.get('away_pitcher_ops', 'N/A'):.3f}")

def tweet_nrfi_probabilities(games, probabilities):
    # Twitter API credentials
    consumer_key = "n3InURKKMBLoViA2PP7EKAHcy"
    consumer_secret = "1AZ23usCQYNMyB4OofeHgPSZ8bHetdKvsZJk8ai1xIVdbnnira"
    access_token = "1650583280255639559-SJLWS2Glxutqx4uDu13ePPmpU0UUdR"
    access_token_secret = "AzpmQeyUP5rsHXXSkYLSxdBVRVDfQfwNaZ3oeITTkujIV"

    # Authenticate to Twitter
    client = tweepy.Client(
        consumer_key=consumer_key, 
        consumer_secret=consumer_secret,
        access_token=access_token, 
        access_token_secret=access_token_secret
    )

    # Tweet each game individually
    for i, game in enumerate(games):
        game_info = (f"NRFI Probability:\n"
                     f"{game['away_team']} @ {game['home_team']}\n"
                     f"Pitchers: {game['away_pitcher']} vs {game['home_pitcher']}\n"
                     f"NRFI Probability: {probabilities[i]:.2%}")
        
        try:
            response = client.create_tweet(text=game_info)
            print(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
        except tweepy.errors.TweepyException as e:
            print(f"Error posting tweet: {e}")

def main():
    # Get current season's start date (assuming it's April 1st, adjust if needed)
    current_year = datetime.now().year
    season_start = datetime(current_year, 4, 1)
    yesterday = datetime.now() - timedelta(days=1)

    # Get season games and stats
    # print(f"Fetching games from {season_start} to {yesterday}")
    season_games = get_season_games(season_start, yesterday)
    team_stats = get_team_stats()
    pitcher_stats = get_pitcher_stats()
    # Prepare historical data for the current season
    historical_data = prepare_data(season_games, team_stats, pitcher_stats)
    
    # Train the model on the current season's data
    model = train_model(historical_data)
    
    # Get today's games and prepare data
    today_games = get_todays_games()
    today_data = prepare_data(today_games, team_stats, pitcher_stats)
    
    # Predict NRFI probabilities for today's games
    probabilities = predict_nrfi_probabilities(model, today_data)
    
    # Create a list of (game, probability) tuples, merging game info with stats
    game_prob_pairs = []
    for i, (_, row) in enumerate(today_data.iterrows()):
        if i in probabilities:
            game_info = row.to_dict()
            game_prob_pairs.append((game_info, probabilities[i]))
        else:
            print(f"Warning: No probability found for game {i}")
    
    # Sort games by NRFI probability
    sorted_games = sorted(game_prob_pairs, key=lambda x: x[1], reverse=True)
    
    print("\n")

    print("\nAll Games Sorted by NRFI Probability:")
    for game, prob in sorted_games:
        print(f"Probability: {prob}")
        print(format_game_info(game, prob))
        print()

    # Tweet all NRFI probabilities
    # tweet_nrfi_probabilities([game for game, _ in sorted_games], 
    #                          [prob for _, prob in sorted_games])

if __name__ == "__main__":
    main()