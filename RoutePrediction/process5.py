import pandas as pd
import networkx as nx

# Read the Excel file into a DataFrame
scats_data = pd.read_excel('Scats Data October 2006.xls', sheet_name='Data', skiprows=1)

# Create a graph
G = nx.Graph()

# Create a dictionary to store intersection data
intersections = {}

# Convert the "Location" column to lowercase before splitting
scats_data['Location'] = scats_data['Location'].str.lower()

# Iterate through the data and extract intersection information
for index, row in scats_data.iterrows():
    location = row['Location']

    # Split the location into road names
    road_names = location.split(" of ")

    # Add the road names as nodes
    for road in road_names:
        G.add_node(road)

    # Add edges between the road names (representing connections)
    if len(road_names) > 1:
        for i in range(len(road_names)):
            for j in range(i + 1, len(road_names)):
                G.add_edge(road_names[i], road_names[j])

# Print debug information
for road_name in G.nodes:
    print(f"Node: {road_name}")
    print(f"Neighbors: {list(G.neighbors(road_name))}")

print("Number of nodes:", len(G.nodes))
print("Number of edges:", len(G.edges))
print("Is the graph connected?", nx.is_connected(G))
