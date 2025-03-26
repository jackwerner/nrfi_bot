import pandas as pd
import os
from datetime import datetime

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