import os

import matplotlib.pyplot as plt
plt.style.use('bmh')
import pandas as pd

def load_training_data(folder_path):
    """Load training data from a CSV log file."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    folder_name = os.path.basename(folder_path)
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df.columns = ['Wall time', 'Step', f'Value_{file}']
        dfs.append(df)

    # Merge all dataframes on 'Step' and not 'Wall time' to align steps across different runs
    merged_df = dfs[0]
    for df in dfs[1:]:
        df = df.drop(columns=['Wall time'])  # Drop 'Wall time' to avoid merge conflicts
        merged_df = pd.merge(merged_df, df, on='Step')
    merged_df.sort_values('Step', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

if __name__ == "__main__":
    folder_path = 'comparison_plots/fpa_50'
    data = load_training_data(folder_path)
    print(data.head())
