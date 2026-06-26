# First batch of libraries
import geopandas as gpd
import json
import jsonlines as jl
import gzip
from pathlib import Path
from tqdm import tqdm

from functools import partial
import networkx as nx
import os
import random
# from typing import Optional

# Libraries GerryChain
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally, cut_edges
from gerrychain.accept import always_accept
from gerrychain.proposals import recom
from gerrychain.constraints import within_percent_of_ideal_population

# required for gerrychain reproducibility
os.environ.setdefault("PYTHONHASHSEED", "0")

def generate_districts(config):
    """
    Run a recom markov chain for each district count
    and write sampled plans to gzipped jsonl files.
    
    Outputs:
        outputs/{run_name}/districts/{run_name}_{n}_districts.jsonl.gz
        Each line: {"assignment": [...], "sample": n}
    """

    # Define seed
    random.seed(config['seed'])

    # Set parameters
    run_name = config["run_name"]
    population_column = config["population_column"]
    chain_length = config["chain_length"]
    n_district = config["district_configs"][0]["num_districts"]
    epsilon = config.get("epsilon", 0.05)

    # Import data
    geodata_path = Path(config["geodata_path"])
    gdf = gpd.read_file(geodata_path)
    
    # Data stats
    print(f"Number of precints: {gdf.shape[0]}\n")
    print(f"Number of columns: {gdf.shape[1]}\n")

    # Transform geopandas to graph object
    graph_path = geodata_path.parent / (
        geodata_path.stem + "_graph.json"
    )
    graph = Graph.from_geodataframe(gdf)
    graph.to_json(str(graph_path))

    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")

    # Quick trick to make sure that if the node labels are not integers, then
    # they are converted to integers starting from 0 so that saving to a JSONL
    # file works correctly every time.
    graph = Graph.from_networkx(
        nx.convert_node_labels_to_integers(graph, first_label=0)
    )
    for node in graph.nodes:
        graph.nodes[node]["poc_vap_20"] = graph.nodes[node]["total_vap_20"] - graph.nodes[node]["white_vap_20"]

    # Step 3 — Output directory
    output_dir = Path(f"outputs/{run_name}/districts")
    output_dir.mkdir(parents=True, exist_ok=True)

    updaters = {
        "population": Tally(population_column,alias = "population"),
        "cut_edges": cut_edges,
        "POCVAP20": Tally("poc_vap_20", "POCVAP20"),
        "VAP20": Tally("total_vap_20", "VAP20"),
        # "PREFERENCE": mmpreference("POCVAP20", "VAP20", pocvap/vap),
        # "MAGNITUDE": lambda P: { d: counter(P["population"][d]) for d in P.parts },
        # "SEATS": TOTAL_SEATS,
        # "STEP": N_ITERATIONS
    }

    # Create an initial partition
    initial_partition = Partition.from_random_assignment(
        graph=graph,
        n_parts=n_district,
        epsilon=epsilon,
        pop_col=population_column,
        updaters=updaters,
        )

    target_population = sum(initial_partition["population"].values()) / n_district

    constraints = [
        within_percent_of_ideal_population(initial_partition, epsilon)
    ]

    recom_proposal = partial(
        recom,
        pop_col=population_column,
        pop_target=target_population,
        epsilon=epsilon,
        node_repeats= 2 # added to prevent stucks at chain
    )

    # Create the Markov chain
    chain = MarkovChain(
        proposal=recom_proposal,
        constraints=constraints,
        accept=always_accept,
        initial_state=initial_partition,
        total_steps=chain_length,
    )

    output_path = (
            output_dir / 
            f"{run_name}_{n_district}_districts.jsonl.gz"
        )
    
    with gzip.open(
            output_path, mode="wt", encoding="utf-8"
        ) as gz_file:
            writer = jl.Writer(gz_file)
            for sample_num, step in enumerate(
                tqdm(
                    chain,
                    total=chain_length,
                    desc=f"{n_district} districts"
                ),
                start=1
            ):
                assignment = list(
                    step.assignment.to_series().sort_index()
                )
                writer.write({
                    "assignment": assignment,
                    "sample": sample_num
                })
            writer.close()