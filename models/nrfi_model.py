import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import pickle
from datetime import datetime
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

def train_model(data):
    # Add runtime tracking
    start_time = datetime.now()
    
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

    # Define the pipeline with preprocessing and model - removed polynomial features
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(objective='binary:logistic', random_state=42))
    ])
    
    # Print information about the features
    print(f"Number of features: {X.shape[1]}")
    
    # Define parameter grid for GridSearchCV
    # param_grid = {
    #     'classifier__n_estimators': [50, 100, 200],
    #     'classifier__max_depth': [3, 4, 5],
    #     'classifier__learning_rate': [0.01, 0.03, 0.05],
    #     'classifier__subsample': [0.7, 0.8, 0.9],
    #     'classifier__colsample_bytree': [0.7, 0.8, 0.9],
    #     'classifier__min_child_weight': [1, 3, 5]
    #     }

    # old params
    param_grid = {
    'classifier__n_estimators': [25, 50, 75, 100, 200, 300],
    'classifier__max_depth': [2, 3, 4, 5],
    'classifier__learning_rate': [0.005, 0.01, 0.03, 0.05],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # Set up GridSearchCV with different scoring metrics
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1, 
        verbose=1,
        return_train_score=True
    )
    
    # Fit the grid search
    print("Starting GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed.")
    
    # Get the best pipeline
    best_pipeline = grid_search.best_estimator_
    
    # Print best parameters
    print("Best parameters:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set with more metrics
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold using various metrics
    thresholds = np.arange(0.4, 0.7, 0.01)  # Expanded range to find better threshold
    results = []
    
    print("\nFinding optimal threshold...")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate predicted positive rate
        pred_pos_rate = np.mean(y_pred)
        
        # Calculate balanced accuracy (average of sensitivity and specificity)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        balanced_acc = (sensitivity + specificity) / 2
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_pos_rate': pred_pos_rate,
            'balanced_acc': balanced_acc
        })
    
    # Convert to DataFrame for easier analysis
    threshold_df = pd.DataFrame(results)
    
    # Find threshold that maximizes balanced accuracy instead of F1
    best_bal_acc_idx = threshold_df['balanced_acc'].idxmax()
    best_threshold = threshold_df.loc[best_bal_acc_idx, 'threshold']
    
    # Print actual class distribution in test set
    actual_pos_rate = y_test.mean()
    print(f"Actual NRFI rate in test set: {actual_pos_rate:.4f}")
    
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Metrics at optimal threshold:")
    print(f"  Accuracy: {threshold_df.loc[best_bal_acc_idx, 'accuracy']:.4f}")
    print(f"  ROC AUC: {threshold_df.loc[best_bal_acc_idx, 'roc_auc']:.4f}")
    print(f"  Precision: {threshold_df.loc[best_bal_acc_idx, 'precision']:.4f}")
    print(f"  Recall: {threshold_df.loc[best_bal_acc_idx, 'recall']:.4f}")
    print(f"  F1 Score: {threshold_df.loc[best_bal_acc_idx, 'f1']:.4f}")
    print(f"  Predicted positive rate: {threshold_df.loc[best_bal_acc_idx, 'pred_pos_rate']:.4f}")
    print(f"  Balanced accuracy: {threshold_df.loc[best_bal_acc_idx, 'balanced_acc']:.4f}")
    
    # Use the best threshold for final predictions
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calculate and print the percentage of predictions that are NRFI (1)
    nrfi_prediction_rate = np.mean(y_pred)
    print(f"Percentage of predictions that are NRFI: {nrfi_prediction_rate:.2%}")

    mse = mean_squared_error(y_test, y_pred_proba)
    print(f"Root mean squared error: {np.sqrt(mse):.4f}")

    print("---Feature importances---")
    classifier = best_pipeline.named_steps['classifier']
    # Get feature importances from XGBoost
    importances = classifier.feature_importances_
    
    # Get feature names (now using original feature names since we removed polynomial features)
    feature_names = X.columns.tolist()
    
    # Display top 20 feature importances
    indices = np.argsort(importances)[::-1]
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

    class_balance = y.value_counts(normalize=True)
    print("Class balance:")
    print(class_balance)

    # Create a figure with multiple subplots for better parameter analysis
    fig = plt.figure(figsize=(20, 15))
    
    # Main grid: 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3)
    
    # First row: Original plots (heatmap and confusion matrix)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot GridSearchCV results - first heatmap (n_estimators vs max_depth)
    results = pd.DataFrame(grid_search.cv_results_)
    pivot_params = ['classifier__n_estimators', 'classifier__max_depth']
    
    pivot = results.pivot_table(
        index=f'param_{pivot_params[0]}', 
        columns=f'param_{pivot_params[1]}',
        values='mean_test_score'
    )
    
    sns.heatmap(pivot, annot=True, cmap='viridis', ax=ax1)
    ax1.set_title(f'GridSearch: {pivot_params[0].split("__")[1]} vs {pivot_params[1].split("__")[1]}')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['RFI', 'NRFI'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix')
    
    # Additional parameter analysis plots
    # Second heatmap: learning_rate vs subsample
    ax3 = fig.add_subplot(gs[0, 2])
    pivot_params2 = ['classifier__learning_rate', 'classifier__subsample']
    pivot2 = results.pivot_table(
        index=f'param_{pivot_params2[0]}', 
        columns=f'param_{pivot_params2[1]}',
        values='mean_test_score'
    )
    sns.heatmap(pivot2, annot=True, cmap='viridis', ax=ax3)
    ax3.set_title(f'GridSearch: {pivot_params2[0].split("__")[1]} vs {pivot_params2[1].split("__")[1]}')
    
    # Third heatmap: learning_rate vs colsample_bytree
    ax4 = fig.add_subplot(gs[1, 0])
    pivot_params3 = ['classifier__learning_rate', 'classifier__colsample_bytree']
    pivot3 = results.pivot_table(
        index=f'param_{pivot_params3[0]}', 
        columns=f'param_{pivot_params3[1]}',
        values='mean_test_score'
    )
    sns.heatmap(pivot3, annot=True, cmap='viridis', ax=ax4)
    ax4.set_title(f'GridSearch: {pivot_params3[0].split("__")[1]} vs {pivot_params3[1].split("__")[1]}')
    
    # Fourth heatmap: subsample vs colsample_bytree
    ax5 = fig.add_subplot(gs[1, 1])
    pivot_params4 = ['classifier__subsample', 'classifier__colsample_bytree']
    pivot4 = results.pivot_table(
        index=f'param_{pivot_params4[0]}', 
        columns=f'param_{pivot_params4[1]}',
        values='mean_test_score'
    )
    
    # Ensure all values are visible in the heatmap
    sns.heatmap(pivot4, annot=True, cmap='viridis', ax=ax5, fmt='.4f')
    ax5.set_title(f'GridSearch: {pivot_params4[0].split("__")[1]} vs {pivot_params4[1].split("__")[1]}')
    
    # Add calibration plot to evaluate probability predictions
    ax6 = fig.add_subplot(gs[1, 2])
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    ax6.plot(prob_pred, prob_true, marker='o', linewidth=1)
    ax6.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
    ax6.set_xlabel('Mean Predicted Probability')
    ax6.set_ylabel('Fraction of Positives')
    ax6.set_title('Calibration Curve')
    
    # Feature importance plot
    ax7 = fig.add_subplot(gs[2, 0])
    # Plot top 15 features or all if less than 15
    n_features = min(15, len(feature_names))
    ax7.bar(range(n_features), importances[indices[:n_features]])
    ax7.set_xticks(range(n_features))
    ax7.set_xticklabels([feature_names[i] for i in indices[:n_features]], rotation=90)
    ax7.set_title('Feature Importances')
    
    # Threshold optimization plot
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(threshold_df['threshold'], threshold_df['accuracy'], label='Accuracy')
    ax8.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision')
    ax8.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall')
    ax8.plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score')
    ax8.plot(threshold_df['threshold'], threshold_df['balanced_acc'], label='Balanced Acc')
    ax8.plot(threshold_df['threshold'], threshold_df['pred_pos_rate'], label='Pred NRFI Rate')
    ax8.axhline(y=actual_pos_rate, color='g', linestyle='--', label=f'Actual NRFI Rate: {actual_pos_rate:.2f}')
    ax8.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
    ax8.set_xlabel('Threshold')
    ax8.set_ylabel('Score')
    ax8.set_title('Metrics vs. Threshold')
    ax8.legend()
    
    # ROC curve
    ax9 = fig.add_subplot(gs[2, 2])
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax9.plot(fpr, tpr)
    ax9.plot([0, 1], [0, 1], 'k--')
    ax9.set_xlabel('False Positive Rate')
    ax9.set_ylabel('True Positive Rate')
    ax9.set_title(f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Add text with model metrics and best parameters
    plt.figtext(0.5, 0.01, 
                f'Best Parameters: {grid_search.best_params_}\n'
                f'Optimal Threshold: {best_threshold:.2f}\n'
                f'Model Metrics: Accuracy: {threshold_df.loc[best_bal_acc_idx, "accuracy"]:.4f}, '
                f'ROC AUC: {threshold_df.loc[best_bal_acc_idx, "roc_auc"]:.4f}, '
                f'F1: {threshold_df.loc[best_bal_acc_idx, "f1"]:.4f}, RMSE: {np.sqrt(mse):.4f}\n'
                f'Class Balance: NRFI={class_balance.get(1, 0):.2%}, RFI={class_balance.get(0, 0):.2%}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plot_file = f"nrfi_model_gridsearch_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model results plot to {plot_file}")

    # Save the trained model and optimal threshold
    model_data = {
        'pipeline': best_pipeline,
        'optimal_threshold': best_threshold
    }
    model_file = f"nrfi_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved trained model to {model_file}")
    
    # Print total runtime
    end_time = datetime.now()
    runtime = end_time - start_time
    print(f"Total model training runtime: {runtime}")

    return model_data

def predict_nrfi_probabilities(model_data, today_data):
    pipeline = model_data['pipeline'] if isinstance(model_data, dict) else model_data
    features = today_data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1, errors='ignore')
    probabilities = pipeline.predict_proba(features)[:, 1]
    return dict(enumerate(probabilities))

def predict_nrfi(model_data, today_data):
    """Predict NRFI using the optimal threshold"""
    if isinstance(model_data, dict) and 'pipeline' in model_data and 'optimal_threshold' in model_data:
        pipeline = model_data['pipeline']
        threshold = model_data['optimal_threshold']
        print(f"Using optimal threshold of {threshold}")
    else:
        # Fallback to default threshold if model_data is just the pipeline
        print("Using default threshold of 0.5")
        pipeline = model_data
        threshold = 0.5
    
    features = today_data.drop(['nrfi', 'home_team', 'away_team', 'home_pitcher', 'away_pitcher'], axis=1, errors='ignore')
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    return dict(enumerate(predictions)), dict(enumerate(probabilities)) 