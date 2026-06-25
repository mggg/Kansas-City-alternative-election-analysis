# First batch of libraries
import geopandas as gpd
import json
import jsonlines
from pathlib import Path

from functools import partial
import networkx as nx
import os
import random
from typing import Optional


# To pass assert
os.environ["PYTHONHASHSEED"] = "0"

# Define Path
BASE_DIR = Path(__file__).parent

# Parameters
BL_PATH = BASE_DIR / "../configs/baseline.json"
with open(BL_PATH) as f:
    parameters = json.load(f)

RUN_NAME = parameters["run_name"]
N_DISTRICTS = [dict_d["num_districts"] for dict_d in parameters["district_configs"]]
print(N_DISTRICTS)

N_ITERATIONS = parameters["chain_length"]
POPULATION_ATTR = parameters["population_column"]
EPSILON = parameters["epsilon"]
RNG_SEED = parameters["seed"]

# print(OUTPUT_FILE)
# TARGET_POPULATION = parameters["target_population"]
# INITIAL_ASSIGNMENT_ATTR = parameters["initial_assignment_attr"]
TOTAL_SEATS = parameters["total_seats"]

# Libraries GerryChain
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally, cut_edges
from gerrychain.accept import always_accept
from gerrychain.proposals import recom
from gerrychain.constraints import within_percent_of_ideal_population

# Missouri VTD Data
gdf_missouri = gpd.read_file("./data/mo_districtr_vtd_view_v1.gpkg",layer="mo_districtr_vtd_view_v1")
districts = gpd.read_file("./data/kcmo_districts.geojson").to_crs(gdf_missouri.crs)

proj_crs = gdf_missouri.estimate_utm_crs()
vtd_rep_points = gdf_missouri.to_crs(proj_crs).copy()
vtd_rep_points["geometry"] = vtd_rep_points.geometry.representative_point()

# Kansas City VTD Data
matched = vtd_rep_points.sjoin(districts.to_crs(proj_crs), how="inner", predicate="within")
gdf = gdf_missouri.loc[matched.index]

# Kansas City VTD Data
# gdf = gdf_missouri.loc[gdf_missouri["path"].str[4:9] == "29095",:]
# print(f"Initial CRS: {gdf.crs}")
# gdf = gdf.to_crs("EPSG:26915")
# print(f"Set CRS: {gdf.crs}")

# Data
print(f"Columns:{gdf.columns.to_list()}\n")
print(f"Number of precints: {gdf.shape[0]}\n")
print(f"Number of columns: {gdf.shape[1]}\n")

# Transform geopandas to graph object
graph = Graph.from_geodataframe(gdf)

# Export
GRAPH_PATH = BASE_DIR / "../outputs/graph/kc_vtd_graph.json"
graph.to_json(GRAPH_PATH)

# print(gdf[["total_vap_20", "white_vap_20"]].head())

# Data
print(f"Number of nodes: {len(graph.nodes)}")
print(f"Number of edges: {len(graph.edges)}")

def run_chain(
    graph_file: str,
    n_districts: int,
    n_iterations: int,
    population_attr: str,
    epsilon: float,
    rng_seed: int,
    output_file: str,
    target_population: Optional[float] = None,
    initial_assignment_attr: Optional[str] = None,
):
    


    graph = Graph.from_json(graph_file)
    random.seed(rng_seed)
    # Need this to be set for reproducibility
    assert os.environ["PYTHONHASHSEED"] == "0"

    # Quick trick to make sure that if the node labels are not integers, then
    # they are converted to integers starting from 0 so that saving to a JSONL
    # file works correctly every time.
    graph = Graph.from_networkx(
        nx.convert_node_labels_to_integers(graph, first_label=0)
    )

    for node in graph.nodes:
        graph.nodes[node]["poc_vap_20"] = graph.nodes[node]["total_vap_20"] - graph.nodes[node]["white_vap_20"]

    # updaters = {
    #     "population": Tally(POPULATION_ATTR, alias="population"),
    # }

    updaters = {
        "population": Tally(population_attr,alias = "population"),
        "cut_edges": cut_edges,
        "POCVAP20": Tally("poc_vap_20", "POCVAP20"),
        "VAP20": Tally("total_vap_20", "VAP20"),
        # "PREFERENCE": mmpreference("POCVAP20", "VAP20", pocvap/vap),
        # "MAGNITUDE": lambda P: { d: counter(P["population"][d]) for d in P.parts },
        # "SEATS": TOTAL_SEATS,
        # "STEP": N_ITERATIONS
    }

    # Create an initial partition
    if initial_assignment_attr is not None:
        initial_partition = Partition(
            graph=graph,
            assignment=initial_assignment_attr,
            updaters=updaters,
        )
    else:
        initial_partition = Partition.from_random_assignment(
            graph=graph,
            n_parts=n_districts,
            epsilon=epsilon,
            pop_col=population_attr,
            updaters=updaters,
        )

    if target_population is None:
        target_population = sum(initial_partition["population"].values()) / len(initial_partition)

    constraints = [
        within_percent_of_ideal_population(initial_partition, epsilon)
    ]

    recom_proposal = partial(
        recom,
        pop_col=population_attr,
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
        total_steps=n_iterations,
    )


    with jsonlines.open(output_file, "w") as writer:
        for i, step in enumerate(chain.with_progress_bar()):
            if step is None:
                continue
            writer.write(
                {
                    "assignment": list(step.assignment.to_series().sort_index()),
                    "sample": i + 1,
                    "poc_vap": {str(d): step["POCVAP20"][d] for d in step.parts},
                    "vap": {str(d): step["VAP20"][d] for d in step.parts},
                    "population": {str(d): step["population"][d] for d in step.parts},
                }
            )


if __name__ == "__main__":

    for n_district in N_DISTRICTS:
        print(f"-------Starting District-size: {n_district}--------")
        output_file = f"../{parameters["gerrychain_output_dir"]}/{RUN_NAME}_{n_district}.json"
        OUTPUT_FILE = BASE_DIR / output_file
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        run_chain(
            graph_file=GRAPH_PATH,
            n_districts=n_district,
            n_iterations=N_ITERATIONS,
            population_attr=POPULATION_ATTR,
            epsilon=EPSILON,
            rng_seed=RNG_SEED,
            output_file=OUTPUT_FILE
        )
        print(f"--------Finishing District-size: {n_district}----------")