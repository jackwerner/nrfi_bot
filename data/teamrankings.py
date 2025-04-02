import requests
from bs4 import BeautifulSoup
import time

def get_teamrankings_stats(team_name, year):
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
        "Oakland Athletics": "Sacramento",  
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
        'nrfi_pct': f'https://www.teamrankings.com/mlb/stat/no-run-first-inning-pct?date={year}-10-31',
        'opponent_nrfi_pct': f'https://www.teamrankings.com/mlb/stat/opponent-no-run-first-inning-pct?date={year}-10-31',
        '1st_inning_runs': f'https://www.teamrankings.com/mlb/stat/1st-inning-runs-per-game?date={year}-10-31',
        'opp_1st_inning_runs': f'https://www.teamrankings.com/mlb/stat/opponent-1st-inning-runs-per-game?date={year}-10-31'
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