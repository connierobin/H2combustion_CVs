import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t', header=None)

def process_data(df, replace_failed=True, max_value=10000):
    if replace_failed:
        df = df.replace(-1, max_value)
    else:
        df = df.replace(-1, np.nan)
    return df

def calculate_statistics(df):
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    return means, stds

def plot_data(datasets, labels):
    result_labels = ['1 (upper right)', '2 (lower left)', '3 (lower right)']

    x = np.arange(len(result_labels))
    width = 0.15  # Width of each bar
    scale_factor = 1.1  # Factor to create gaps between groups

    fig, ax = plt.subplots()
    
    for i, (means, stds) in enumerate(datasets):
        ax.bar(x * scale_factor + i * width - (width * len(datasets) / 2), means, width, label=labels[i], yerr=stds, capsize=5)


    ax.set_xlabel('Wells')
    ax.set_ylabel('Average Time Step')
    ax.set_title('First Time Step To Wolfe Schlegel Quandrant')
    ax.set_xticks(x)
    ax.set_xticklabels(result_labels)
    ax.legend()

    plt.show()

def compare_PCA_sigmas():
    # List of files to read
    file_paths = ['PCA-3.txt', 'PCA-2.txt', 'PCA-1.txt', 'PCA-4.txt']
    labels = ['PCA sigma=0.5', 'PCA sigma=0.6', 'PCA sigma=0.7', 'PCA sigma=0.8']
    
    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def AE_vs_PCA_sigmas():
    # List of files to read
    file_paths = ['AE-1.txt', 'PCA-3.txt', 'PCA-2.txt', 'PCA-1.txt', 'PCA-4.txt']
    labels = ['AE', 'PCA sigma=0.5', 'PCA sigma=0.6', 'PCA sigma=0.7', 'PCA sigma=0.8']
    
    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def AE_vs_PCA_sigmas_noreplace():
    # List of files to read
    file_paths = ['AE-1.txt', 'PCA-3.txt', 'PCA-2.txt', 'PCA-1.txt', 'PCA-4.txt']
    labels = ['AE Filtered', 'PCA sigma=0.5', 'PCA sigma=0.6', 'PCA sigma=0.7', 'PCA sigma=0.8']
    
    # Process data from all files
    datasets = []
    replace_failed = False  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def compare_PCA_ns():
    # List of files to read
    file_paths = ['PCA-1.txt', 'PCA-7.txt', 'PCA-6.txt', 'PCA-5.txt']
    labels = ['PCA n=0', 'PCA n=1', 'PCA n=2', 'PCA n=3']
    
    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def compare_AE_ns():
    # List of files to read
    file_paths = ['AE-1.txt', 'AE-2.txt']
    labels = ['AE n=0', 'AE n=3']

    
    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def compare_AE_ns_noreplace():
    # List of files to read
    file_paths = ['AE-1.txt', 'AE-2.txt']
    labels = ['AE n=0 Filtered', 'AE n=3 Filtered']

    
    # Process data from all files
    datasets = []
    replace_failed = False  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def AE_vs_PCA_ns():
    file_paths = ['AE-1.txt', 'AE-2.txt', 'PCA-1.txt', 'PCA-7.txt', 'PCA-6.txt', 'PCA-5.txt']
    labels = ['AE n=0', 'AE n=3', 'PCA n=0', 'PCA n=1', 'PCA n=2', 'PCA n=3']

    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def compare_AE_activation_fns():
    # List of files to read
    file_paths = ['AE-1.txt', 'AE-3.txt', 'AE-4.txt', 'AE-5.txt']
    labels = ['AE ReLU', 'AE tanh', 'AE leaky ReLU', 'AE elu']

    
    # Process data from all files
    datasets = []
    replace_failed = True  # Change to False to exclude failed results

    for file_path in file_paths:
        df = read_data(file_path)
        df_processed = process_data(df, replace_failed)
        means, stds = calculate_statistics(df_processed)
        datasets.append((means, stds))

    # Plot data
    plot_data(datasets, labels)

def main():
    compare_PCA_sigmas()
    AE_vs_PCA_sigmas()
    AE_vs_PCA_sigmas_noreplace()
    compare_PCA_ns()
    compare_AE_ns()
    compare_AE_ns_noreplace()
    # AE_vs_PCA_ns()
    compare_AE_activation_fns()

if __name__ == "__main__":
    main()
