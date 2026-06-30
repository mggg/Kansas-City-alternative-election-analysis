# Ensemble of Redistricting Plans using Recom

Why Ensemble Redistricting? Because we are going to create thousands of pausible legally-valid plans of a same-size district.

Steps of the process:

1. Transform geodata to dual graph.
- We represent Kansas City as a graph where each node corresponds to a precint/ block, joined by an edge for adjancent units. This is also a districting plan where we have an assignment of every node to a district number. The adjacent nodes are geographically adjacent precints/ blocks.

2. Initial Random Partition
- Recursive tree splitting algorithm defines a starting plan. In other words, we are going to generate a concrete assignment of precints/blocks into districts that satifies population balance rules. So, MC have an input to where to start its random wlaks