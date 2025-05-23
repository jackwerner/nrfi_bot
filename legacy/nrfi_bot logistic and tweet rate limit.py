import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tweepy
import numpy as np
import json
import time
from datetime import datetime, timedelta

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
                strikeOuts = stats_data.get('strikeOuts')
                plateAppearances = stats_data.get('plateAppearances')
                
                if runs is not None and games is not None and games > 0:
                    runs_scored_per_game = runs / games
                else:
                    runs_scored_per_game = None
                    print(f"Unable to calculate runsScoredPerGame for team {team_name} (ID: {team_id})")
                
                team_stats[team_id] = {
                    'name': team_name,
                    'runsScoredPerGame': runs_scored_per_game,
                    'obp': stats_data.get('obp'),
                    'slg': stats_data.get('slg'),
                    'strikeOutRate': strikeOuts / plateAppearances if strikeOuts and plateAppearances else None,
                    'babip': stats_data.get('babip'),
                    'atBatsPerHomeRun': stats_data.get('atBatsPerHomeRun')
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
        "playerPool": "All",
        "limit": 100,
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
                    'obp': split['stat'].get('obp'),
                    'slg': split['stat'].get('slg'),
                    'strikeoutWalkRatio': split['stat'].get('strikeoutWalkRatio'),
                    'homeRunsPer9': split['stat'].get('homeRunsPer9')
                }
                for stat, value in pitcher_stats[pitcher_name].items():
                    if value is None:
                        print(f"Missing {stat} for pitcher {pitcher_name}")
            
            if len(data['stats'][0]['splits']) < params['limit']:
                break
            
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
        
        if home_pitcher not in pitcher_stats:
            missing_pitchers.add(home_pitcher)
            excluded_games += 1
            continue
        if away_pitcher not in pitcher_stats:
            missing_pitchers.add(away_pitcher)
            excluded_games += 1
            continue
        
        if home_team_id in team_stats and away_team_id in team_stats:
            home_team_data = team_stats[home_team_id]
            away_team_data = team_stats[away_team_id]
            
            game_data = {
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_pitcher': home_pitcher,
                'away_pitcher': away_pitcher,
                'home_runsScoredPerGame': home_team_data['runsScoredPerGame'],
                'home_obp': home_team_data['obp'],
                'home_slg': home_team_data['slg'],
                'home_strikeOutRate': home_team_data['strikeOutRate'],
                'home_babip': home_team_data['babip'],
                'home_atBatsPerHomeRun': home_team_data['atBatsPerHomeRun'],
                'away_runsScoredPerGame': away_team_data['runsScoredPerGame'],
                'away_obp': away_team_data['obp'],
                'away_slg': away_team_data['slg'],
                'away_strikeOutRate': away_team_data['strikeOutRate'],
                'away_babip': away_team_data['babip'],
                'away_atBatsPerHomeRun': away_team_data['atBatsPerHomeRun'],
                'home_pitcher_era': pitcher_stats[home_pitcher]['era'],
                'home_pitcher_whip': pitcher_stats[home_pitcher]['whip'],
                'home_pitcher_strikeoutsPer9Inn': pitcher_stats[home_pitcher]['strikeoutsPer9Inn'],
                'home_pitcher_obp': pitcher_stats[home_pitcher]['obp'],
                'home_pitcher_slg': pitcher_stats[home_pitcher]['slg'],
                'home_pitcher_strikeoutWalkRatio': pitcher_stats[home_pitcher]['strikeoutWalkRatio'],
                'home_pitcher_homeRunsPer9': pitcher_stats[home_pitcher]['homeRunsPer9'],
                'away_pitcher_era': pitcher_stats[away_pitcher]['era'],
                'away_pitcher_whip': pitcher_stats[away_pitcher]['whip'],
                'away_pitcher_strikeoutsPer9Inn': pitcher_stats[away_pitcher]['strikeoutsPer9Inn'],
                'away_pitcher_obp': pitcher_stats[away_pitcher]['obp'],
                'away_pitcher_slg': pitcher_stats[away_pitcher]['slg'],
                'away_pitcher_strikeoutWalkRatio': pitcher_stats[away_pitcher]['strikeoutWalkRatio'],
                'away_pitcher_homeRunsPer9': pitcher_stats[away_pitcher]['homeRunsPer9']
            }
            
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
    
    for col in df.columns:
        if col not in ['home_team', 'away_team', 'home_pitcher', 'away_pitcher']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    return df

def train_model(data):
    X = data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1)
    y = data['nrfi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = pipeline.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    r2 = r2_score(y_test, y_pred_proba)
    print(f"R-squared score: {r2:.2f}")

    mse = mean_squared_error(y_test, y_pred_proba)
    print(f"Mean squared error: {mse:.4f}")

    print("---Feature coefficients---")
    classifier = pipeline.named_steps['classifier']
    for feature, coefficient in zip(X.columns, classifier.coef_[0]):
        print(f"{feature}: {coefficient:.4f}")

    class_balance = y.value_counts(normalize=True)
    print("Class balance:")
    print(class_balance)

    return pipeline

def predict_nrfi_probabilities(pipeline, today_data):
    features = today_data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1, errors='ignore')
    probabilities = pipeline.predict_proba(features)[:, 1]
    return dict(enumerate(probabilities))

def format_game_info(game, probability):
    
    return (f"{game['away_team']} @ {game['home_team']}\n"
            f"Pitchers: {game['away_pitcher']} vs {game['home_pitcher']}\n"
            f"NRFI Probability: {probability:.2%}\n"
            f"Home Team Stats:\n"
            f"  Runs Scored Per Game: {game.get('home_runsScoredPerGame', 'N/A'):.4f}\n"
            f"  OBP: {game.get('home_obp', 'N/A'):.3f}\n"
            f"  SLG: {game.get('home_slg', 'N/A'):.3f}\n"
            f"  Strike Out Rate: {game.get('home_strikeOutRate', 'N/A'):.3f}\n"
            f"  BABIP: {game.get('home_babip', 'N/A'):.3f}\n"
            f"  At Bats Per Home Run: {game.get('home_atBatsPerHomeRun', 'N/A'):.3f}\n"
            f"Away Team Stats:\n"
            f"  Runs Scored Per Game: {game.get('away_runsScoredPerGame', 'N/A'):.4f}\n"
            f"  OBP: {game.get('away_obp', 'N/A'):.3f}\n"
            f"  SLG: {game.get('away_slg', 'N/A'):.3f}\n"
            f"  Strike Out Rate: {game.get('away_strikeOutRate', 'N/A'):.3f}\n"
            f"  BABIP: {game.get('away_babip', 'N/A'):.3f}\n"
            f"  At Bats Per Home Run: {game.get('away_atBatsPerHomeRun', 'N/A'):.3f}\n"
            f"Home Pitcher Stats:\n"
            f"  ERA: {game.get('home_pitcher_era', 'N/A'):.2f}\n"
            f"  WHIP: {game.get('home_pitcher_whip', 'N/A'):.2f}\n"
            f"  K/9: {game.get('home_pitcher_strikeoutsPer9Inn', 'N/A'):.2f}\n"
            f"  OBP Against: {game.get('home_pitcher_obp', 'N/A'):.3f}\n"
            f"  SLG Against: {game.get('home_pitcher_slg', 'N/A'):.3f}\n"
            f"  Strikeout Walk Ratio: {game.get('home_pitcher_strikeoutWalkRatio', 'N/A'):.3f}\n"
            f"  Home Runs Per 9: {game.get('home_pitcher_homeRunsPer9', 'N/A'):.3f}\n"
            f"Away Pitcher Stats:\n"
            f"  ERA: {game.get('away_pitcher_era', 'N/A'):.2f}\n"
            f"  WHIP: {game.get('away_pitcher_whip', 'N/A'):.2f}\n"
            f"  K/9: {game.get('away_pitcher_strikeoutsPer9Inn', 'N/A'):.2f}\n"
            f"  OBP Against: {game.get('away_pitcher_obp', 'N/A'):.3f}\n"
            f"  SLG Against: {game.get('away_pitcher_slg', 'N/A'):.3f}\n"
            f"  Strikeout Walk Ratio: {game.get('away_pitcher_strikeoutWalkRatio', 'N/A'):.3f}\n"
            f"  Home Runs Per 9: {game.get('away_pitcher_homeRunsPer9', 'N/A'):.3f}")

def tweet_nrfi_probabilities(games, probabilities):
    consumer_key = "n3InURKKMBLoViA2PP7EKAHcy"
    consumer_secret = "1AZ23usCQYNMyB4OofeHgPSZ8bHetdKvsZJk8ai1xIVdbnnira"
    access_token = "1650583280255639559-SJLWS2Glxutqx4uDu13ePPmpU0UUdR"
    access_token_secret = "AzpmQeyUP5rsHXXSkYLSxdBVRVDfQfwNaZ3oeITTkujIV"

    client = tweepy.Client(
        consumer_key=consumer_key, 
        consumer_secret=consumer_secret,
        access_token=access_token, 
        access_token_secret=access_token_secret
    )

    tweets_sent = 0

    for i, game in enumerate(games):
        game_info = (f"⚾ NRFI Probability ⚾\n"
                     f"{game['away_team']} @ {game['home_team']}\n"
                     f"Pitchers: {game['away_pitcher']} 🆚 {game['home_pitcher']}\n"
                     f"NRFI Probability: {probabilities[i]:.2%} {'📈' if probabilities[i] > 0.5 else '📉' if probabilities[i] < 0.5 else '⚖️'}")
        
        while True:
            try:
                response = client.create_tweet(text=game_info)
                tweets_sent += 1
                print(f"Tweet posted successfully! Tweet ID: {response.data['id']}")

                # Check rate limit status
                limit = int(response.headers['x-rate-limit-limit'])
                remaining = int(response.headers['x-rate-limit-remaining'])
                reset_time = int(response.headers['x-rate-limit-reset'])

                print(f"Rate limit status: {remaining}/{limit}")
                
                if remaining <= 1:
                    reset_datetime = datetime.fromtimestamp(reset_time)
                    wait_time = (reset_datetime - datetime.now()).total_seconds() + 1
                    print(f"Rate limit nearly exhausted. Waiting until {reset_datetime} ({wait_time:.0f} seconds)")
                    time.sleep(wait_time)
                else:
                    time.sleep(5)  # Small delay between tweets to be safe

                break

            except tweepy.errors.TooManyRequests as e:
                reset_time = int(e.response.headers['x-rate-limit-reset'])
                reset_datetime = datetime.fromtimestamp(reset_time)
                wait_time = (reset_datetime - datetime.now()).total_seconds() + 1
                print(f"Rate limit exceeded. Waiting until {reset_datetime} ({wait_time:.0f} seconds)")
                time.sleep(wait_time)

            except tweepy.errors.TweepyException as e:
                print(f"Error posting tweet: {e}")
                break

    print(f"Total tweets sent: {tweets_sent}")
    return tweets_sent

def main():
    current_year = datetime.now().year
    season_start = datetime(current_year, 4, 1)
    yesterday = datetime.now() - timedelta(days=1)

    season_games = get_season_games(season_start, yesterday)
    team_stats = get_team_stats()
    pitcher_stats = get_pitcher_stats()
    historical_data = prepare_data(season_games, team_stats, pitcher_stats)
    
    model = train_model(historical_data)
    
    today_games = get_todays_games()
    today_data = prepare_data(today_games, team_stats, pitcher_stats)
    
    probabilities = predict_nrfi_probabilities(model, today_data)
    
    game_prob_pairs = []
    for i, (_, row) in enumerate(today_data.iterrows()):
        if i in probabilities:
            game_info = row.to_dict()
            game_prob_pairs.append((game_info, probabilities[i]))
        else:
            print(f"Warning: No probability found for game {i}")
    
    sorted_games = sorted(game_prob_pairs, key=lambda x: x[1], reverse=True)
    
    print("\n")

    print("\nAll Games Sorted by NRFI Probability:")
    for game, prob in sorted_games:
        print(f"Probability: {prob}")
        print(format_game_info(game, prob))
        print()

    # Tweet all NRFI probabilities
    tweets_sent = tweet_nrfi_probabilities([game for game, _ in sorted_games], 
                                           [prob for _, prob in sorted_games])
    print(f"Total tweets sent in this run: {tweets_sent}")

if __name__ == "__main__":
    main()