import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_plots(df, y_test, y_pred, labels):
    # 1. Pairplot (Feature Relationships)
    sns.set(style="ticks")
    pair_plot = sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
    pair_plot.savefig('outputs/iris_pairplot.png')
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model Performance')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    print("Visualizations saved to outputs/ folder.")