import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
print("plt:",plt)
print("plt.figure:", plt.figure)
def boxplot(df, number, stat):
    """
    df : pd.DataFrame with columns Chordal_AMD_true, Chordal_AMD_false, Chordal_MFI_true, Chordal_MFI_false and profile
    number: int, the number of the plot
    
    For each column get the all the values and plot all the values in a boxplot where every strategy is a different box.
    Title of the Plot should be the number of the plot and the number of profiles in the df and written as "Solve time per strategy {number} (sorted by {stat} ) - N = {len(df)}"
    """
    #Strategies are the columns of the df that start with "Chordal_"
    strategies = [c for c in df.columns if c.startswith("Chordal_")]

    # Prepare data for boxplot
    data = [df[strategy].dropna() for strategy in strategies]
    
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=strategies)
    plt.title(f"Solve time per strategy {number} (sorted by {stat}) - N = {len(df)}")
    plt.ylabel('Solve Time')
    plt.xlabel('Decomposition Strategy')
    plt.grid(True)
    plt.show()

def sort_by_plot(df, sort_by, number_of_plots):
    """ df: pd.DataFrame with the the statistics of each profile
        sort_by: str, the column to sort by
        number_of_plots: int, the number of subplots in the subplots
        
        The functions splits the df in number_of_plots, based on the sort_by column
        Then, each subdataset is named by its order in the sort_by column
        Then we boxplot the SolveTime per decomp strategy for each subdataset"""
        
    # Sort the DataFrame by the specified column
    sorted_df = df.sort_values(by=sort_by, ascending=False)
    print(sorted_df.head())
    
    # Split the DataFrame into chunks for plotting
    chunk_size = len(sorted_df) // number_of_plots
    chunks = [sorted_df.iloc[i:i + chunk_size] for i in range(0, len(sorted_df), chunk_size)]
    
    # If the last chunk is smaller than chunk_size, we merge it with the previous one
    if len(chunks) > number_of_plots:
        chunks[-2] = pd.concat([chunks[-2], chunks[-1]], ignore_index=True)
        chunks = chunks[:-1]
    
    # Assign names to each chunk based on the sort_by column
    for i, chunk in enumerate(chunks):
        chunk_name = f"{sort_by}_{i+1}"
        chunk['chunk_name'] = chunk_name
        boxplot(chunk, i, sort_by)
        
        
    #Create the boxplots for each chunk
        
        
    return chunks
      
def sort_by_plot_subplots(df: pd.DataFrame,
                          sort_by: str,
                          number_of_plots: int, name:str = "Solve_time"):
    """
    Splits `df` into `number_of_plots` chunks sorted by `sort_by`, then
    makes one figure with `number_of_plots` side-by-side boxplots of 
    solve times per decomposition strategy.
    """
    # 1) Sort once
    sorted_df = df.sort_values(by=sort_by, ascending=False)
    N = len(sorted_df)

    # 2) Split into chunks
    chunk_size = N // number_of_plots
    chunks = []
    for i in range(number_of_plots):
        start = i * chunk_size
        end   = (i + 1) * chunk_size if i < number_of_plots - 1 else N
        chunks.append(sorted_df.iloc[start:end])
    
    # 3) Detect your four strategy columns dynamically:
    strategies = [c for c in df.columns if c.startswith("Chordal_")]

    strategies = sorted(strategies)  # alphabetical: ['AMD_false','AMD_true',…]
    labels     = strategies[:]       # same order for labels

    # 4) Make subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=number_of_plots,
        figsize=(5 * number_of_plots, 5),
        sharey=True
    )
    if number_of_plots == 1:
        axes = [axes]

    # 5) Fill each
    for i, (ax, chunk) in enumerate(zip(axes, chunks)):
        data = [chunk[strat].dropna() for strat in strategies]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{sort_by}_{i+1}\nN={len(chunk)}")
        ax.set_xlabel("Decomp Strategy")
        if i == 0:
            ax.set_ylabel(name)

    # 6) Super‐title & layout
    fig.suptitle(f"{name} per strategy (sorted by {sort_by}) — Total N={N}",
                 y=1.02)
    plt.tight_layout()
    plt.show()

    return chunks

def plot_confusion_matrix(save_dir, y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_dir = save_dir + "/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Saved confusion matrix → {save_path}")
    plt.show()

def plot_regret_boxplot(save_dir,Y_test, preds, strategy_labels=None):
    Y_preds = [Y_test[j, preds[j]] for j in range(len(preds))]
    Y_test_copy = np.hstack((Y_test, np.array(Y_preds).reshape(-1, 1)))

    if strategy_labels is None:
        strategy_labels = [f"{i}" for i in range(Y_test.shape[1])] + ["Model"]

    plt.figure(figsize=(12, 6))
    plt.boxplot(Y_test_copy, labels=strategy_labels)
    plt.title("Boxplot of Regrets per Decomposition Strategy")
    plt.xlabel("Decomposition Strategy")
    plt.ylabel("Regret")
    plt.grid(True)
    plt.tight_layout()
    save_dir = save_dir + "/boxplots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"boxplot.png")
    plt.savefig(save_path)
    print(f"Saved boxplot → {save_path}")
    plt.show()

    return Y_test_copy