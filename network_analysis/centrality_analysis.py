import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os


def compute_centrality_measures(graphml_path, partition="material"):
    """
    Load bipartite network from .graphml and compute centrality measures for the given partition.
    Returns a DataFrame of centrality values for nodes in the selected partition.
    """
    # Load graph
    G = nx.read_graphml(graphml_path)

    # Extract nodes in specified partition
    nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == partition]
    if not nodes:
        raise ValueError(f"No nodes found in partition '{partition}'.")

    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coefficient = nx.clustering(G)

    # Collect values for selected nodes
    data = {
        "node": nodes,
        "degree": [G.degree(n) for n in nodes],
        "degree_centrality": [degree_centrality[n] for n in nodes],
        "closeness": [closeness_centrality[n] for n in nodes],
        "betweenness": [betweenness_centrality[n] for n in nodes],
        "clustering": [clustering_coefficient[n] for n in nodes],
    }

    return pd.DataFrame(data)


def plot_and_save_histogram(data, column, xlabel, output_dir):
    """
    Create and save histogram for a given centrality measure.
    """
    plt.figure(figsize=(8, 4))
    if column == "degree":
        bins = range(data[column].min(), data[column].max() + 1)
        plt.hist(data[column], bins=bins, edgecolor="black", color="gray", align="left")
    else:
        plt.hist(data[column], bins=50, edgecolor="black", color="skyblue")

    plt.xlabel(xlabel)
    plt.ylabel("Number of Nodes")
    plt.title(f"{xlabel} Distribution")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{column}_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


# NEW (minimal change): make your "main" configurable for notebook use
def run(
    graphml_path="/home/anoj/work/network/bine/data/all_materials/all_materials_network.graphml",
    output_dir="centrality_outputs",
    partition="motif",
    save_csv=True,
    make_plots=False,
):
    # === Create output directory ===
    os.makedirs(output_dir, exist_ok=True)

    # === Compute centralities ===
    centrality_df = compute_centrality_measures(graphml_path, partition=partition)

    # === Save data to CSV ===
    csv_path = None
    if save_csv:
        csv_path = os.path.join(output_dir, f"{partition}_centrality_data.csv")
        centrality_df.to_csv(csv_path, index=False)

    # === Plot and save histograms (optional) ===
    if make_plots:
        plot_and_save_histogram(centrality_df, "degree", "Degree", output_dir)
        plot_and_save_histogram(centrality_df, "betweenness", "Betweenness Centrality", output_dir)
        plot_and_save_histogram(centrality_df, "closeness", "Closeness Centrality", output_dir)
        plot_and_save_histogram(centrality_df, "clustering", "Clustering Coefficient", output_dir)

    print(f"Centrality analysis completed. Data saved in: {output_dir}")
    if csv_path:
        print(f"CSV: {csv_path}")

    # Return df so notebook can use it directly
    return centrality_df


def main():
    # === Settings ===
    graphml_path = "../data/all_materials/all_materials_network.graphml"
    output_dir = "centrality_outputs"
    partition = "material"

    # This keeps original workflow identical:
    run(
        graphml_path=graphml_path,
        output_dir=output_dir,
        partition=partition,
        save_csv=True,
        make_plots=False,   # same as your commented plots block
    )


if __name__ == "__main__":
    main()
