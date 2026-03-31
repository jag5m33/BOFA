import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, average_precision_score
from pybofa.prep.config import data as dcfg

# 1. SETUP & LOADING
plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv(dcfg.final_results)
y_true = (df['source'] == 'GH_CONTROL').astype(int)

def plot_consolidated_performance(df, y_true):
    """Generates a clean Ensemble ROC and PR Curve with model variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    score_cols = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    
    # --- ROC CURVE ---
    # Plot individual models as faint lines
    for col in score_cols:
        fpr, tpr, _ = roc_curve(y_true, df[col])
        ax1.plot(fpr, tpr, color='grey', lw=1, alpha=0.2)

    # Plot the Total (Weighted Ensemble) Score
    fpr_en, tpr_en, _ = roc_curve(y_true, df['total_score'])
    roc_auc_en = auc(fpr_en, tpr_en)
    ax1.plot(fpr_en, tpr_en, color='#d62728', lw=3, label=f'Ensemble System (AUC = {roc_auc_en:.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title('Overall System ROC', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')

    # --- PRECISION-RECALL CURVE ---
    # Plot individual models as faint lines
    for col in score_cols:
        precision, recall, _ = precision_recall_curve(y_true, df[col])
        ax2.plot(recall, precision, color='grey', lw=1, alpha=0.2)

    # Plot the Total (Weighted Ensemble) Score
    precision_en, recall_en, _ = precision_recall_curve(y_true, df['total_score'])
    ap_en = average_precision_score(y_true, df['total_score'])
    ax2.plot(recall_en, precision_en, color='#2ca02c', lw=3, label=f'Ensemble System (AP = {ap_en:.2f})')

    ax2.set_title('Overall Precision-Recall', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (Positive Predictive Value)')
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('overall_performance_clean.png', dpi=300)

def plot_confusion_matrix(df, y_true, target_recall=0.70):
    """Visualizes the Hits and False Alarms."""
    doped_scores = df[df['source'] == 'GH_CONTROL']['total_score']
    thresh = np.percentile(doped_scores, (1 - target_recall) * 100)
    y_pred = (df['total_score'] >= thresh).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Ref (Clean)', 'GH (Doped)'], 
                yticklabels=['Ref (Clean)', 'GH (Doped)'])
    
    plt.title(f'Confusion Matrix @ {target_recall*100}% Recall\n(System Threshold: {thresh:.2f})', fontweight='bold')
    plt.ylabel('Actual State')
    plt.xlabel('Detection Decision')
    plt.savefig('confusion_matrix_clean.png', dpi=300)

def visualize_model_mechanics(df):
    """Visualizes individual model distributions for the Methods section."""
    models = ['ae_score', 'if_score', 'gmm_score', 'svm_score']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, m in enumerate(models):
        sns.kdeplot(data=df, x=m, hue='source', fill=True, ax=axes[i], common_norm=False, palette='muted')
        axes[i].set_title(f'Algorithm Density: {m.split("_")[0].upper()}', fontweight='bold')
        axes[i].set_xlabel('Standardized Anomaly Score')

    plt.suptitle('Algorithm Logic: Distribution of Anomaly Scores by Population', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('algorithm_mechanics.png', dpi=300)

if __name__ == "__main__":
    print("Generating Clean Dissertation Visuals...")
    
    # 1. Plot consolidated ROC/PR (Faint grey for individuals, bold for average)
    plot_consolidated_performance(df, y_true)
    
    # 2. Confusion Matrix
    plot_confusion_matrix(df, y_true)
    
    # 3. Algorithm distributions (Great for showing how they work)
    visualize_model_mechanics(df)
    
    print("Done! Files saved: overall_performance_clean.png, confusion_matrix_clean.png, algorithm_mechanics.png")