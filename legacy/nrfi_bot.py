import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tweepy
import numpy as np
import json
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import os
import dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

dotenv.load_dotenv()

def get_todays_games():
    print(f"Getting today's games")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        # "date": datetime.now().strftime("%Y-%m-%d"),
        "date": "2024-05-15",
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
                    game_time = datetime.strptime(game['gameDate'], "%Y-%m-%dT%H:%M:%SZ")
                    game_time_est = game_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-4)))
                    games.append({
                        'game_id': game['gamePk'],
                        'home_team': game['teams']['home']['team']['name'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_team_id': game['teams']['home']['team']['id'],
                        'away_team_id': game['teams']['away']['team']['id'],
                        'home_pitcher': home_pitcher,
                        'away_pitcher': away_pitcher,
                        'game_time': game_time_est
                    })
        return games
    else:
        raise Exception(f"Failed to fetch games: {response.status_code}")

def get_team_stats():
    # Check if cached data exists
    print(f"Getting team stats for {datetime.now().year}")
    cache_file = f"team_stats_{datetime.now().year}.json"
    if os.path.exists(cache_file):
        print(f"Loading cached team stats from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    url = "https://statsapi.mlb.com/api/v1/teams"
    params = {
        "sportId": 1,
        "season": 2024 #datetime.now().year
    }
    response = requests.get(url, params=params)
    print(f"Team list API call: Status code {response.status_code}")
    if response.status_code == 200:
        teams = response.json()['teams']
        team_stats = {}
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            stats_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season=2024"#{datetime.now().year}"
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
                
                # Add additional NRFI-related stats from TeamRankings
                team_stats[team_id].update(get_teamrankings_stats(team_name))
                
                for stat, value in team_stats[team_id].items():
                    if value is None:
                        print(f"Missing {stat} for team {team_name}")
            else:
                print(f"Failed to fetch stats for {team_name}: Status code {stats_response.status_code}")
        
        # Save to cache file
        with open(cache_file, 'w') as f:
            json.dump(team_stats, f)
        print(f"Saved team stats to {cache_file}")
        
        return team_stats
    else:
        raise Exception(f"Failed to fetch team stats: {response.status_code}")

def get_teamrankings_stats(team_name):
    """
    Get NRFI-related stats from TeamRankings.com for a specific team
    """
    print(f"Getting TeamRankings stats for {team_name}")
    # Map MLB API team names to TeamRankings team names
    team_name_mapping = {
        "Arizona Diamondbacks": "Arizona",
        "Atlanta Braves": "Atlanta",
        "Baltimore Orioles": "Baltimore",
        "Boston Red Sox": "Boston",
        "Chicago Cubs": "Chi Cubs",
        "Chicago White Sox": "Chi Sox",
        "Cincinnati Reds": "Cincinnati",
        "Cleveland Guardians": "Cleveland",
        "Colorado Rockies": "Colorado",
        "Detroit Tigers": "Detroit",
        "Houston Astros": "Houston",
        "Kansas City Royals": "Kansas City",
        "Los Angeles Angels": "LA Angels",
        "Los Angeles Dodgers": "LA Dodgers",
        "Miami Marlins": "Miami",
        "Milwaukee Brewers": "Milwaukee",
        "Minnesota Twins": "Minnesota",
        "New York Mets": "NY Mets",
        "New York Yankees": "NY Yankees",
        "Athletics": "Sacramento",  # Oakland is now Sacramento in TeamRankings
        "Philadelphia Phillies": "Philadelphia",
        "Pittsburgh Pirates": "Pittsburgh",
        "San Diego Padres": "San Diego",
        "San Francisco Giants": "SF Giants",
        "Seattle Mariners": "Seattle",
        "St. Louis Cardinals": "St. Louis",
        "Tampa Bay Rays": "Tampa Bay",
        "Texas Rangers": "Texas",
        "Toronto Blue Jays": "Toronto",
        "Washington Nationals": "Washington"
    }
    
    # Convert team name to TeamRankings format
    tr_team_name = team_name_mapping.get(team_name, team_name)
    
    # URLs for the stats we want to scrape
    urls = {
        'nrfi_pct': 'https://www.teamrankings.com/mlb/stat/no-run-first-inning-pct?date=2024-10-31',
        'opponent_nrfi_pct': 'https://www.teamrankings.com/mlb/stat/opponent-no-run-first-inning-pct?date=2024-10-31',
        '1st_inning_runs': 'https://www.teamrankings.com/mlb/stat/1st-inning-runs-per-game?date=2024-10-31',
        'opp_1st_inning_runs': 'https://www.teamrankings.com/mlb/stat/opponent-1st-inning-runs-per-game?date=2024-10-31'
    }
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    team_stats = {}
    
    for stat_name, url in urls.items():
        try:
            # Add a delay to avoid overloading the server
            time.sleep(1)
            
            # Send request and get HTML content
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table
            table = soup.find('table', {'class': 'tr-table datatable'})
            if not table:
                table = soup.find('table')
            
            if not table:
                print(f"Table not found for {stat_name}")
                continue
            
            # Find the row for the team
            team_row = None
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if cells and len(cells) > 1:
                    row_team_name = cells[1].text.strip()
                    if row_team_name == tr_team_name:
                        team_row = cells
                        break
            
            if team_row and len(team_row) > 2:
                # Get the current season value (usually in the 3rd column)
                value_text = team_row[2].text.strip()
                try:
                    # Convert percentage to decimal if needed
                    if '%' in value_text:
                        value = float(value_text.replace('%', '')) / 100
                    else:
                        value = float(value_text)
                    team_stats[stat_name] = value
                except ValueError:
                    print(f"Could not convert {value_text} to float for {stat_name}")
                    team_stats[stat_name] = None
            else:
                print(f"Team {tr_team_name} not found in {stat_name} table")
                team_stats[stat_name] = None
                
        except Exception as e:
            print(f"Error scraping {stat_name} for {tr_team_name}: {e}")
            team_stats[stat_name] = None
    
    return team_stats

def get_pitcher_stats():
    # Check if cached data exists
    print(f"Getting pitcher stats for {datetime.now().year}")
    cache_file = f"pitcher_stats_{datetime.now().year}.json"
    if os.path.exists(cache_file):
        print(f"Loading cached pitcher stats from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season",
        "group": "pitching",
        "season": 2024,#datetime.now().year,
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
    
    # Save to cache file
    with open(cache_file, 'w') as f:
        json.dump(pitcher_stats, f)
    print(f"Saved pitcher stats to {cache_file}")
    
    return pitcher_stats

def get_season_games(start_date, end_date):
    # Check if cached data exists
    cache_file = f"season_games_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    if os.path.exists(cache_file):
        print(f"Loading cached season games from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
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
                    # print(game)
                    if 'linescore' in game and game['linescore']['innings']:
                        # print("linescore found")
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
    
    # Save to cache file
    with open(cache_file, 'w') as f:
        json.dump(games, f)
    print(f"Saved season games to {cache_file}")
    
    return games

def prepare_data(games, team_stats, pitcher_stats):
    # Check if prepared_data.csv exists and return it if so
    csv_file = "prepared_data.csv"
    if os.path.exists(csv_file):
        print(f"Loading prepared data from existing {csv_file}")
        return pd.read_csv(csv_file)
    
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
            
            # Add the new TeamRankings stats if available
            for stat in ['nrfi_pct', 'opponent_nrfi_pct', '1st_inning_runs', 'opp_1st_inning_runs']:
                if stat in home_team_data:
                    game_data[f'home_{stat}'] = home_team_data[stat]
                if stat in away_team_data:
                    game_data[f'away_{stat}'] = away_team_data[stat]
            
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
    
    # Save the prepared data to CSV
    if 'nrfi' in df.columns:  # Only save historical data, not prediction data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"prepared_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved prepared data to {csv_file}")
    
    return df

def train_model(data):
    # Check if required columns exist before dropping them
    columns_to_drop = []
    for col in ['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher']:
        if col in data.columns:
            columns_to_drop.append(col)
    
    X = data.drop(columns_to_drop, axis=1)
    
    # If 'nrfi' column doesn't exist, we can't train the model
    if 'nrfi' not in data.columns:
        print("Error: 'nrfi' column not found in data. Cannot train model.")
        print("Available columns:", data.columns.tolist())
        return None
    
    y = data['nrfi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = pipeline.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    r2 = r2_score(y_test, y_pred_proba)
    print(f"R-squared score: {r2:.2f}")

    mse = mean_squared_error(y_test, y_pred_proba)
    print(f"Root mean squared error: {np.sqrt(mse):.4f}")

    print("---Feature importances---")
    classifier = pipeline.named_steps['classifier']
    # Get feature importances from XGBoost instead of coefficients
    importances = classifier.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.4f}")

    class_balance = y.value_counts(normalize=True)
    print("Class balance:")
    print(class_balance)

    # Create and save a scatterplot of predicted probabilities vs actual outcomes
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot of predicted probabilities
    ax1.scatter(y_pred_proba, y_test, alpha=0.5)
    ax1.set_xlabel('Predicted NRFI Probability')
    ax1.set_ylabel('Actual Outcome (1=NRFI, 0=RFI)')
    ax1.set_title(f'NRFI Prediction Scatter Plot\nAccuracy: {accuracy:.2f}, R²: {r2:.2f}')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0.5 to show the decision boundary
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    # Add a diagonal line for perfect predictions
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['RFI', 'NRFI'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix')
    
    # Add text with model metrics
    plt.figtext(0.5, 0.01, 
                f'Model Metrics:\nAccuracy: {accuracy:.2f}, R²: {r2:.2f}, MSE: {mse:.4f}\n'
                f'Class Balance: NRFI={class_balance.get(1, 0):.2%}, RFI={class_balance.get(0, 0):.2%}',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plot_file = f"nrfi_model_results_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model results plot to {plot_file}")

    # Save the trained model
    model_file = f"nrfi_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_file, 'wb') as f:
        import pickle
        pickle.dump(pipeline, f)
    print(f"Saved trained model to {model_file}")

    return pipeline

def predict_nrfi_probabilities(pipeline, today_data):
    features = today_data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1, errors='ignore')
    probabilities = pipeline.predict_proba(features)[:, 1]
    return dict(enumerate(probabilities))

def format_game_info(game, probability):
    game_time_str = game['game_time'].strftime("%I:%M %p ET") if 'game_time' in game else "Time N/A"
    result = (f"{game['away_team']} @ {game['home_team']} - {game_time_str}\n"
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
    
    # Add the new TeamRankings stats if available
    additional_stats = []
    for prefix, team in [('home', 'Home'), ('away', 'Away')]:
        for stat, label in [
            ('nrfi_pct', 'NRFI %'), 
            ('opponent_nrfi_pct', 'Opponent NRFI %'),
            ('1st_inning_runs', '1st Inning Runs/Game'),
            ('opp_1st_inning_runs', 'Opponent 1st Inning Runs/Game')
        ]:
            key = f"{prefix}_{stat}"
            if key in game:
                if 'pct' in stat:
                    additional_stats.append(f"  {team} {label}: {game.get(key, 'N/A'):.1%}")
                else:
                    additional_stats.append(f"  {team} {label}: {game.get(key, 'N/A'):.2f}")
    
    if additional_stats:
        result += "\nTeamRankings Stats:\n" + "\n".join(additional_stats)
    
    return result

def tweet_nrfi_probabilities(games, probabilities):
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
        game_info = (f"⚾ NRFI Probability ⚾\n"
                     f"{game['away_team']} @ {game['home_team']} - {game_time_str}⏰\n"
                     f"Pitchers: {game['away_pitcher']} 🆚 {game['home_pitcher']}\n"
                     f"NRFI Probability: {probabilities[i]:.2%} {'📈' if probabilities[i] > 0.5 else '📉' if probabilities[i] < 0.5 else '⚖️'}")
        
        while True:
            try:
                response = client.create_tweet(text=game_info)
                tweets_sent += 1
                print(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
                time.sleep(10)
                # # Check rate limit status
                # limit = int(response.headers['x-rate-limit-limit'])
                # remaining = int(response.headers['x-rate-limit-remaining'])
                # reset_time = int(response.headers['x-rate-limit-reset'])

                # print(f"Rate limit status: {remaining}/{limit}")
                
                # if remaining <= 1:
                #     reset_datetime = datetime.fromtimestamp(reset_time)
                #     wait_time = (reset_datetime - datetime.now()).total_seconds() + 1
                #     print(f"Rate limit nearly exhausted. Waiting until {reset_datetime} ({wait_time:.0f} seconds)")
                #     time.sleep(wait_time)
                # else:
                #     time.sleep(5)  # Small delay between tweets to be safe

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

def main():
    # current_year = datetime.now().year
    # season_start = datetime(current_year, 4, 1)
    # yesterday = datetime.now() - timedelta(days=1)
    season_start = datetime(2024, 4, 1)
    current_year = 2024
    yesterday = datetime(2024, 9, 30)

    season_games = get_season_games(season_start, yesterday)
    team_stats = get_team_stats()
    pitcher_stats = get_pitcher_stats()
    historical_data = prepare_data(season_games, team_stats, pitcher_stats)
    
    model = train_model(historical_data)
    
    today_games = get_todays_games()
    today_data = prepare_data(today_games, team_stats, pitcher_stats)
    
    probabilities = predict_nrfi_probabilities(model, today_data)
    
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
    
    sorted_games = sorted(game_prob_pairs, key=lambda x: x[1], reverse=True)
    
    print("\nAll Games Sorted by NRFI Probability:")
    for game, prob in sorted_games:
        print(f"Probability: {prob}")
        print(format_game_info(game, prob))
        print()

    tweets_sent = tweet_nrfi_probabilities([game for game, _ in sorted_games], 
                                           [prob for _, prob in sorted_games])
    print(f"Total tweets sent in this run: {tweets_sent}")

if __name__ == "__main__":
    main()