import requests
from datetime import datetime, timedelta, timezone
import json
import os
import time
from data.teamrankings import get_teamrankings_stats

def get_todays_games():
    print(f"Getting today's games")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        # "date": "2024-05-15",
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
        "season": 2025, #datetime.now().year,
        "gameType": "S", # change to use R for regular season 
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