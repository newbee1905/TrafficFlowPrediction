import pandas as pd
import networkx as nx

# Read the Excel file into a DataFrame, skipping the first row (times row)
scats_data = pd.read_excel('Scats Data October 2006.xls', sheet_name='Data', skiprows=1)

# Sort the data by 'Location' and 'Date' columns to ensure consistency
scats_data = scats_data.sort_values(by=['Location', 'Date'])

# Create a graph
road_network = nx.Graph()

# Create a dictionary to store the intersections connected by each road segment
road_segment_intersections = {}

# Create a dictionary to store the number of road segments connected to each intersection
intersection_counts = {}

# Iterate through the data and add intersections as nodes
for index, row in scats_data.iterrows():
    intersection_name = row['Location']
    latitude = row['NB_LATITUDE']
    longitude = row['NB_LONGITUDE']

    # Extract the road segments by splitting the "Location" string using " of " as the delimiter
    road_segment_parts = intersection_name.split(' of ')

    # If there are exactly two parts, this is a physically connected intersection
    if len(road_segment_parts) == 2:
        road_segment = tuple(sorted(road_segment_parts))  # Sort the parts to ensure consistency

        # Add the intersections to the graph if they're not already present
        for intersection in road_segment_parts:
            if intersection not in road_network:
                road_network.add_node(intersection, latitude=latitude, longitude=longitude)
                intersection_counts[intersection] = 0  # Initialize intersection count

        # Store the intersections connected by this road segment
        if road_segment in road_segment_intersections:
            road_segment_intersections[road_segment].append(road_segment_parts)
        else:
            road_segment_intersections[road_segment] = [road_segment_parts]

        # Increment the count for each intersection
        for intersection in road_segment_parts:
            intersection_counts[intersection] += 1

# Create road segments based on the intersections they connect
for road_segment, intersections in road_segment_intersections.items():
    road_name = f"{intersections[0][0]} - {intersections[1][0]}"  # Use the first parts of the intersections for the road name
    road_network.add_edge(intersections[0][0], intersections[1][0], road_name=road_name)

# Check if each intersection connects to exactly four road segments
for intersection, count in intersection_counts.items():
    if count != 4:
        print(f"Intersection '{intersection}' connects to {count} road segments. This does not match the expected 4 connections.")

# Print the number of nodes and edges in the road network
num_nodes = road_network.number_of_nodes()
num_edges = road_network.number_of_edges()
print(f"Number of nodes in the graph: {num_nodes}")
print(f"Number of edges in the graph: {num_edges}")
