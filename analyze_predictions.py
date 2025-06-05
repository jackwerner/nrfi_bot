import os
import pandas as pd
import glob
from datetime import datetime, timedelta
from data.mlb_api import get_season_games

def analyze_predictions(start_date=None, end_date=None):
    """
    Analyze NRFI predictions by comparing them with actual game results.
    
    Args:
        start_date (datetime, optional): Start date for analysis
        end_date (datetime, optional): End date for analysis
    
    Returns:
        dict: Analysis results
    """
    # Find all prediction files
    prediction_files = glob.glob('predictions/nrfi_predictions_*.csv')
    
    if not prediction_files:
        print("No prediction files found in predictions/ directory")
        return None
    
    # Get today's date to exclude today's predictions
    today = datetime.now().date()
    
    # Filter files by date range if provided
    filtered_files = []
    for file in prediction_files:
        date_str = file.split('_')[-1].replace('.csv', '')
        file_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Skip today's predictions
        if file_date.date() == today:
            print(f"Skipping today's predictions: {file}")
            continue
            
        if start_date and file_date.date() < start_date.date():
            continue
        if end_date and file_date.date() > end_date.date():
            continue
        
        filtered_files.append(file)
    
    prediction_files = filtered_files
    
    # Sort files by date
    prediction_files.sort()
    
    if not prediction_files:
        print("No prediction files found for the specified date range")
        return None
    
    print(f"Analyzing {len(prediction_files)} prediction files")
    
    # Initialize results tracking
    total_predictions = 0
    correct_predictions = 0
    nrfi_predictions = 0
    correct_nrfi = 0
    yrfi_predictions = 0
    correct_yrfi = 0
    
    # Process each file
    for file in prediction_files:
        print(f"Processing {file}")
        predictions_df = pd.read_csv(file)
        
        # Extract date from filename
        date_str = file.split('_')[-1].replace('.csv', '')
        file_date = datetime.strptime(date_str, '%Y-%m-%d')
        print(f"Processing {file_date}")
        # Get actual game results
        actual_games = get_season_games(file_date, file_date)
        
        # Match predictions with actual results
        for _, prediction in predictions_df.iterrows():
            # Find matching game in actual_games
            matching_game = None
            for game in actual_games:
                if (prediction['home_team'] == game['home_team'] and 
                    prediction['away_team'] == game['away_team'] and
                    prediction['home_pitcher'] == game['home_pitcher'] and
                    prediction['away_pitcher'] == game['away_pitcher']):
                    matching_game = game
                    break
            
            if matching_game:
                total_predictions += 1
                
                # Check if prediction was correct
                predicted_nrfi = prediction['prediction'] == 'NRFI'
                actual_nrfi = matching_game['nrfi'] == 1
                
                if predicted_nrfi:
                    nrfi_predictions += 1
                    if actual_nrfi:
                        correct_nrfi += 1
                        correct_predictions += 1
                else:
                    yrfi_predictions += 1
                    if not actual_nrfi:
                        correct_yrfi += 1
                        correct_predictions += 1
            else:
                print(f"Warning: Could not find matching game for {prediction['away_team']} @ {prediction['home_team']}")
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    nrfi_accuracy = correct_nrfi / nrfi_predictions if nrfi_predictions > 0 else 0
    yrfi_accuracy = correct_yrfi / yrfi_predictions if yrfi_predictions > 0 else 0
    
    # Prepare results
    results = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': overall_accuracy,
        'nrfi_predictions': nrfi_predictions,
        'correct_nrfi': correct_nrfi,
        'nrfi_accuracy': nrfi_accuracy,
        'yrfi_predictions': yrfi_predictions,
        'correct_yrfi': correct_yrfi,
        'yrfi_accuracy': yrfi_accuracy
    }
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Total Predictions: {total_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"NRFI Predictions: {nrfi_predictions}, Correct: {correct_nrfi}, Accuracy: {nrfi_accuracy:.2%}")
    print(f"YRFI Predictions: {yrfi_predictions}, Correct: {correct_yrfi}, Accuracy: {yrfi_accuracy:.2%}")
    
    return results

def detailed_analysis(start_date=None, end_date=None):
    """Generate a detailed analysis with game-by-game results"""
    # Find all prediction files
    prediction_files = glob.glob('predictions/nrfi_predictions_*.csv')
    
    if not prediction_files:
        print("No prediction files found in predictions/ directory")
        return None
    
    # Get today's date to exclude today's predictions
    today = datetime.now().date()
    
    # Filter files by date range if provided
    filtered_files = []
    for file in prediction_files:
        date_str = file.split('_')[-1].replace('.csv', '')
        file_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Skip today's predictions
        if file_date.date() == today:
            print(f"Skipping today's predictions: {file}")
            continue
            
        if start_date and file_date.date() < start_date.date():
            continue
        if end_date and file_date.date() > end_date.date():
            continue
        
        filtered_files.append(file)
    
    prediction_files = filtered_files
    
    # Sort files by date
    prediction_files.sort()
    
    if not prediction_files:
        print("No prediction files found for the specified date range")
        return None
    
    # Initialize results dataframe
    results = []
    
    # Process each file
    for file in prediction_files:
        print(f"Processing {file} for detailed analysis")
        predictions_df = pd.read_csv(file)
        
        # Extract date from filename
        date_str = file.split('_')[-1].replace('.csv', '')
        file_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get actual game results
        actual_games = get_season_games(file_date, file_date)
        
        # Match predictions with actual results
        for _, prediction in predictions_df.iterrows():
            # Find matching game in actual_games
            matching_game = None
            for game in actual_games:
                if (prediction['home_team'] == game['home_team'] and 
                    prediction['away_team'] == game['away_team'] and
                    prediction['home_pitcher'] == game['home_pitcher'] and
                    prediction['away_pitcher'] == game['away_pitcher']):
                    matching_game = game
                    break
            
            if matching_game:
                predicted_nrfi = prediction['prediction'] == 'NRFI'
                actual_nrfi = matching_game['nrfi'] == 1
                correct = (predicted_nrfi and actual_nrfi) or (not predicted_nrfi and not actual_nrfi)
                
                results.append({
                    'date': prediction['date'],
                    'away_team': prediction['away_team'],
                    'home_team': prediction['home_team'],
                    'away_pitcher': prediction['away_pitcher'],
                    'home_pitcher': prediction['home_pitcher'],
                    'predicted': 'NRFI' if predicted_nrfi else 'YRFI',
                    'actual': 'NRFI' if actual_nrfi else 'YRFI',
                    'correct': correct,
                    'nrfi_probability': prediction['nrfi_probability'],
                    'threshold': prediction['threshold']
                })
            else:
                print(f"Warning: Could not find matching game for {prediction['away_team']} @ {prediction['home_team']}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_file = 'prediction_analysis.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to {results_file}")
        
        # Calculate threshold quality
        threshold_analysis = analyze_threshold_quality(results_df)
        
        return {
            'detailed_results': results_df,
            'threshold_analysis': threshold_analysis
        }
    
    return None

def analyze_threshold_quality(results_df):
    """Analyze how well the probability threshold performed"""
    # Group results by threshold
    threshold_groups = results_df.groupby('threshold')
    
    threshold_metrics = []
    
    for threshold, group in threshold_groups:
        nrfi_predictions = group[group['predicted'] == 'NRFI']
        nrfi_correct = nrfi_predictions[nrfi_predictions['correct'] == True]
        
        nrfi_accuracy = len(nrfi_correct) / len(nrfi_predictions) if len(nrfi_predictions) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'total_predictions': len(group),
            'nrfi_predictions': len(nrfi_predictions),
            'nrfi_correct': len(nrfi_correct),
            'nrfi_accuracy': nrfi_accuracy
        })
    
    return pd.DataFrame(threshold_metrics)

def pretty_analysis(start_date=None, end_date=None):
    """
    Generate a visually appealing presentation of prediction analysis results
    
    Args:
        start_date (datetime, optional): Start date for analysis
        end_date (datetime, optional): End date for analysis
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        from rich.text import Text
    except ImportError:
        print("Please install rich library: pip install rich")
        return
    
    console = Console()
    
    # Get analysis results
    console.print("[bold blue]Running prediction analysis...[/bold blue]")
    results = analyze_predictions(start_date, end_date)
    detailed = detailed_analysis(start_date, end_date)
    
    if not results or not detailed:
        console.print("[bold red]No prediction data available for analysis[/bold red]")
        return
    
    # Create header
    console.print(Panel.fit(
        Text("NRFI Prediction Performance Analysis", style="bold white"),
        border_style="blue",
        box=box.DOUBLE
    ))
    
    # Overall stats table
    stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_column("Percentage", justify="right")
    
    stats_table.add_row(
        "Total Predictions", 
        str(results['total_predictions']),
        ""
    )
    stats_table.add_row(
        "Correct Predictions", 
        str(results['correct_predictions']),
        f"{results['overall_accuracy']:.2%}"
    )
    stats_table.add_row(
        "NRFI Predictions", 
        str(results['nrfi_predictions']),
        f"{results['nrfi_accuracy']:.2%}"
    )
    stats_table.add_row(
        "YRFI Predictions", 
        str(results['yrfi_predictions']),
        f"{results['yrfi_accuracy']:.2%}"
    )
    
    console.print(Panel(stats_table, title="Overall Statistics"))
    
    # Threshold analysis
    if 'threshold_analysis' in detailed:
        threshold_df = detailed['threshold_analysis']
        
        threshold_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        threshold_table.add_column("Threshold")
        threshold_table.add_column("Predictions", justify="right")
        threshold_table.add_column("NRFI Predictions", justify="right")
        threshold_table.add_column("Accuracy", justify="right")
        
        for _, row in threshold_df.iterrows():
            accuracy_color = "green" if row['nrfi_accuracy'] >= 0.6 else "red"
            threshold_table.add_row(
                str(row['threshold']),
                str(row['total_predictions']),
                str(row['nrfi_predictions']),
                f"[{accuracy_color}]{row['nrfi_accuracy']:.2%}[/{accuracy_color}]"
            )
        
        console.print(Panel(threshold_table, title="Threshold Performance"))
    
    # Recent performance (last 10 games)
    if 'detailed_results' in detailed and not detailed['detailed_results'].empty:
        recent_df = detailed['detailed_results'].sort_values('date', ascending=False).head(10)
        
        recent_table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
        recent_table.add_column("Date")
        recent_table.add_column("Matchup")
        recent_table.add_column("Prediction")
        recent_table.add_column("Actual")
        recent_table.add_column("Result")
        
        for _, row in recent_df.iterrows():
            result_color = "green" if row['correct'] else "red"
            result_text = "✓" if row['correct'] else "✗"
            
            recent_table.add_row(
                row['date'],
                f"{row['away_team']} @ {row['home_team']}",
                row['predicted'],
                row['actual'],
                f"[{result_color}]{result_text}[/{result_color}]"
            )
        
        console.print(Panel(recent_table, title="Recent Predictions (Last 10)"))
    
    # Print recommendations
    if results['nrfi_accuracy'] > results['yrfi_accuracy']:
        recommendation = "NRFI predictions are performing better than YRFI predictions"
    else:
        recommendation = "YRFI predictions are performing better than NRFI predictions"
    
    console.print(Panel(
        f"[bold]{recommendation}[/bold]\n\n" +
        f"Overall model accuracy: [{'green' if results['overall_accuracy'] >= 0.55 else 'red'}]{results['overall_accuracy']:.2%}[/]",
        title="Insights",
        border_style="green"
    ))

if __name__ == "__main__":
    print("NRFI Prediction Analysis")
    print("-----------------------")
    
    # Generate pretty analysis
    pretty_analysis()
    
    # You can also analyze specific date ranges:
    # start_date = datetime(2024, 6, 1)
    # end_date = datetime(2024, 6, 30)
    # pretty_analysis(start_date, end_date)
    
    # Or use the original analysis functions
    # analyze_predictions()
    # detailed_analysis()
