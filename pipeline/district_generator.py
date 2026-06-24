# First batch of libraries
import geopandas as gpd
import json
import numpy as np
import jsonlines
from pathlib import Path

# import click
from functools import partial
import networkx as nx
import os
import random
from typing import Optional

# Define Path
BASE_DIR = Path(__file__).parent

# Parameters
BL_PATH = BASE_DIR / "../configs/baseline.json"
with open(BL_PATH) as f:
    parameters = json.load(f)

RUN_NAME = parameters["run_name"]
N_DISTRICTS = parameters["district_configs"]["num_districts"]
N_ITERATIONS = parameters["chain_length"]
POPULATION_ATTR = parameters["population_column"]
EPSILON = parameters["epsilon"]
RNG_SEED = parameters["seed"]
OUTPUT_FILE = BASE_DIR / f"../{parameters["gerrychain_output_dir"]}/{RUN_NAME}.json"
TARGET_POPULATION = parameters["target_population"]
INITIAL_ASSIGNMENT_ATTR = parameters["initial_assignment_attr"]

TOTAL_SEATS = parameters["total_seats"]


# Libraries GerryChain
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally, cut_edges
from gerrychain.accept import always_accept
from gerrychain.proposals import recom

# Missouri VTD Data
gdf_missouri = gpd.read_file("./inputs/mo_districtr_vtd_view_v1.gpkg",layer="mo_districtr_vtd_view_v1")
# Kansas City VTD Data
gdf = gdf_missouri.loc[gdf_missouri["path"].str[4:9] == "29095",:]

# Data
print(f"Columns:{gdf.columns.to_list()}\n")
print(f"Number of precints: {gdf.shape[0]}\n")
print(f"Number of columns: {gdf.shape[1]}\n")

# Transform geopandas to graph object
graph = Graph.from_geodataframe(gdf) # crs_override, how do I define the CRS?

# Data
print(f"Number of nodes: {len(graph.nodes)}")
print(f"Number of edges: {len(graph.edges)}")

# def run_chain(
#     graph_file: str,
#     n_districts: int,
#     n_iterations: int,
#     population_attr: str,
#     epsilon: float,
#     rng_seed: int,
#     output_file: str,
#     target_population: Optional[float] = None,
#     initial_assignment_attr: Optional[str] = None,
# ):

random.seed(RNG_SEED)

# Need this to be set for reproducibility
assert os.environ["PYTHONHASHSEED"] == "0"

# Quick trick to make sure that if the node labels are not integers, then
# they are converted to integers starting from 0 so that saving to a JSONL
# file works correctly every time.
graph = Graph.from_networkx(
    nx.convert_node_labels_to_integers(graph, first_label=0)
)

updaters = {
    "population": Tally(POPULATION_ATTR, alias="population"),
}
updaters = {
    "population": Tally(POPULATION_ATTR,alias = "population"),
    "cut_edges": cut_edges,
    "POCVAP20": Tally("POCVAP20", "POCVAP20"),
    "VAP20": Tally("VAP20", "VAP20"),
    # "PREFERENCE": mmpreference("POCVAP20", "VAP20", pocvap/vap),
    # "MAGNITUDE": lambda P: { d: counter(P["population"][d]) for d in P.parts },
    "SEATS": TOTAL_SEATS,
    "STEP": N_ITERATIONS
}

# Create an initial partition
if INITIAL_ASSIGNMENT_ATTR is not None:
    initial_partition = Partition(
        graph=graph,
        assignment=INITIAL_ASSIGNMENT_ATTR,
        updaters=updaters,
    )
else:
    initial_partition = Partition.from_random_assignment(
        graph=graph,
        n_parts=N_DISTRICTS,
        epsilon=EPSILON,
        pop_col=POPULATION_ATTR,
        updaters=updaters,
    )

if TARGET_POPULATION is None:
    target_population = sum(initial_partition["population"].values()) / N_DISTRICTS

recom_proposal = partial(
    recom,
    pop_col=POPULATION_ATTR,
    pop_target=target_population,
    epsilon=EPSILON
)

# Crete the Markov chain
chain = MarkovChain(
    proposal=recom_proposal,
    constraints=[],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=N_ITERATIONS,
)

with jsonlines.open(OUTPUT_FILE, "w") as writer:
    for i, step in enumerate(chain):
        if step is None:
            continue
        writer.write(
            {
                "assignment": list(step.assignment.to_series().sort_index()),
                "sample": i + 1,
            }
        )


# @click.command()
# @click.option(
#     "--graph-file",
#     type=click.Path(exists=True),
#     required=True,
#     help="Path to the graph file.",
# )
# @click.option(
#     "--n-districts", type=int, required=True, help="Number of districts to create."
# )
# @click.option(
#     "--n-iterations",
#     type=int,
#     default=1000,
#     help="Number of iterations for the Markov chain.",
# )
# @click.option(
#     "--population-attr",
#     type=str,
#     default="population",
#     help="Population attribute for the nodes.",
# )
# @click.option(
#     "--target-population", type=float, help="Target population for each district."
# )
# @click.option(
#     "--epsilon", type=float, default=0.01, help="Epsilon value for the Markov chain."
# )
# @click.option(
#     "--rng-seed", type=int, default=42, help="Random seed for reproducibility."
# )
# @click.option(
#     "--output-file", type=click.Path(), required=True, help="Path to save the output."
# )
# @click.option(
#     "--initial_assignment_attr",
#     type=str,
#     default=None,
#     help="Initial assignment attribute for the nodes.",
# )
# def run_chain_cli(
#     graph_file: str,
#     n_districts: int,
#     n_iterations: int,
#     population_attr: str,
#     target_population: float,
#     epsilon: float,
#     rng_seed: int,
#     output_file: str,
#     initial_assignment_attr: Optional[str] = None,
# ):
# run_chain(
#         graph_file=graph_file,
#         n_districts=n_districts,
#         n_iterations=n_iterations,
#         population_attr=population_attr,
#         target_population=target_population,
#         epsilon=epsilon,
#         rng_seed=rng_seed,
#         output_file=output_file,
#         initial_assignment_attr=initial_assignment_attr,
# )


# if __name__ == "__main__":
#     run_chain_cli()