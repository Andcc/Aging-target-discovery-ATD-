import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import networkx as nx
import pickle as pkl
from datetime import datetime


def setup_logger():
    """Setup logger that outputs to console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # File Handler
    log_directory = '/mnt/e/projects/aging_target_discovery/logs/'
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    log_filename = f"{current_time}_centrality_feature_calculation.log"
    log_filepath = os.path.join(log_directory, log_filename)

    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def read_network_data():
    """Read network data from preprocessed STRING csv file."""
    data_dir = '/mnt/e/projects/aging_target_discovery/data'
    data_filepath = os.path.join(data_dir, 'processed_string_hDf.csv')

    try:
        return pd.read_csv(data_filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {data_filepath}")
        return None


def calculate_centrality(G, centrality_type):
    """Calculate the specified centrality metric for the graph."""
    logging.info(f"Calculating {centrality_type} centrality...")
    if centrality_type == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'eigenvector':
        centrality = nx.eigenvector_centrality(G)
    logging.info(f"Finished calculating {centrality_type} centrality.")
    return centrality


def load_or_compute_centrality(G, centrality_type):
    """Load centrality if file exists, else compute and save."""
    filename = f"/mnt/e/projects/aging_target_discovery/data/intermediate_data/{centrality_type}_centrality_stringNetwork.pkl"

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            logging.info(f"Loading {centrality_type} centrality from file...")
            return pkl.load(f)
    else:
        logging.info(f"Calculating {centrality_type} centrality...")
        centrality = calculate_centrality(G, centrality_type)
        logging.info(f"Finished calculating {centrality_type} centrality.")

        with open(filename, 'wb') as f:
            pkl.dump(centrality, f)
        logging.info(f"Finished saving {centrality_type} centrality to file.")

        return centrality


def main():
    setup_logger()

    # Read network data
    logging.info("Reading network data...")
    string_human_df_processed = read_network_data()
    if string_human_df_processed is None:
        return
    logging.info("Finished reading network data.")

    # Create network graph
    logging.info("Creating network graph...")
    G = nx.from_pandas_edgelist(
        string_human_df_processed, 'protein1', 'protein2', edge_attr=True)
    logging.info("Finished creating network graph.")

    # Calculate/load centrality measures
    degree_centrality = load_or_compute_centrality(G, 'degree')
    closeness_centrality = load_or_compute_centrality(G, 'closeness')
    betweenness_centrality = load_or_compute_centrality(G, 'betweenness')
    eigenvector_centrality = load_or_compute_centrality(G, 'eigenvector')

    # Calculate clustering coefficient
    logging.info("Calculating clustering coefficients...")
    clustering_coefficient = nx.clustering(G)
    logging.info("Finished calculating clustering coefficients.")

    # Convert to dataframes and merge
    logging.info("Converting to DataFrames and merging...")
    degree_df = pd.DataFrame(degree_centrality.items(), columns=[
                             'protein', 'degree_centrality'])
    closeness_df = pd.DataFrame(closeness_centrality.items(), columns=[
                                'protein', 'closeness_centrality'])
    betweenness_df = pd.DataFrame(betweenness_centrality.items(), columns=[
                                  'protein', 'betweenness_centrality'])
    eigenvector_df = pd.DataFrame(eigenvector_centrality.items(), columns=[
                                  'protein', 'eigenvector_centrality'])
    clustering_df = pd.DataFrame(clustering_coefficient.items(), columns=[
                                 'protein', 'clustering_coefficient'])

    feature_df = pd.concat([degree_df, closeness_df['closeness_centrality'],
                            betweenness_df['betweenness_centrality'], eigenvector_df['eigenvector_centrality'],
                            clustering_df['clustering_coefficient']], axis=1)
    logging.info("Finished converting to DataFrames and merging.")

    # Save data & display summary statistics
    with open('../data/intermediate_data/centrality_feature_df.pkl', 'wb') as f:
        logging.info("Saving feature DataFrame to file...")
        pkl.dump(feature_df, f)
    logging.info("Finished saving feature DataFrame to file.")

    # Summary statistics
    logging.info("Calculating summary statistics...")
    summary_stats = feature_df.describe()
    logging.info("Finished calculating summary statistics.")
    logging.info(f"\n{summary_stats}")

    # Correlation matrix
    logging.info("Calculating correlation matrix...")
    correlation_matrix = feature_df.corr()
    logging.info("Finished calculating correlation matrix.")
    logging.info(f"\n{correlation_matrix}")

    # Plot centrality distributions
    import matplotlib.pyplot as plt

    logging.info("Plotting centrality distributions...")
    plt.figure(figsize=(12, 6))
    plt.hist(degree_df['degree_centrality'], bins=20,
             alpha=0.5, label='Degree Centrality')
    plt.hist(closeness_df['closeness_centrality'],
             bins=20, alpha=0.5, label='Closeness Centrality')
    plt.hist(betweenness_df['betweenness_centrality'],
             bins=20, alpha=0.5, label='Betweenness Centrality')
    plt.hist(eigenvector_df['eigenvector_centrality'],
             bins=20, alpha=0.5, label='Eigenvector Centrality')
    plt.xlabel('Centrality')
    plt.ylabel('Frequency')
    plt.title('Centrality Distributions')
    plt.legend()
    plt.savefig('centrality_distributions.png')
    plt.close()
    logging.info("Finished plotting centrality distributions.")

    # Plot correlation matrix
    logging.info("Plotting correlation matrix...")
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix)),
               correlation_matrix.columns, rotation='vertical')
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title('Centrality Correlation Matrix')
    plt.savefig('centrality_correlation_matrix.png')
    plt.close()
    logging.info("Finished plotting correlation matrix.")


if __name__ == '__main__':
    main()
