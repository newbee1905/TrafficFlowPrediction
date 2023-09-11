import pandas as pd
import networkx as nx
from math import atan2, radians, sin, cos, sqrt

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula to calculate distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

# Read the Excel file into a DataFrame
scats_data = pd.read_excel('Scats Data October 2006.xls', sheet_name='Data', skiprows=1)

# Create a graph
G = nx.Graph()

# Create a dictionary to store intersection data
intersections = {}

# Convert the "Location" column to lowercase before splitting
scats_data['Location'] = scats_data['Location'].str.lower()

# Iterate through the data and extract intersection information
# Iterate through the data and extract intersection information
for index, row in scats_data.iterrows():
    location = row['Location']

    # Split the location into road names
    road_names = location.split(" of ")

    # Add the road names as nodes
    for road in road_names:
        road_lower = road.lower()
        if road_lower not in G:
            G.add_node(road_lower, original_name=road)  # Convert to lowercase and store original case

        # Add the road names to the intersections dictionary if not already added
        if road_lower not in intersections:
            intersections[road_lower] = {
                'original_name': road,
                'latitude': row['NB_LATITUDE'],
                'longitude': row['NB_LONGITUDE']
            }

    # Add edges between the road names (representing connections)
    if len(road_names) > 1:
        for i in range(len(road_names)):
            for j in range(i + 1, len(road_names)):
                road1_lower = road_names[i].lower()
                road2_lower = road_names[j].lower()
                if G.has_node(road1_lower) and G.has_node(road2_lower) and not G.has_edge(road1_lower, road2_lower):
                    lat1, lon1 = intersections[road1_lower]['latitude'], intersections[road1_lower]['longitude']
                    lat2, lon2 = intersections[road2_lower]['latitude'], intersections[road2_lower]['longitude']
                    distance = haversine(lat1, lon1, lat2, lon2)  # Use the correct latitudes and longitudes
                    G.add_edge(road1_lower, road2_lower, weight=distance)

# Debug information
for node in G.nodes:
    print(f"Node: {node}")
    print(f"Original Name: {G.nodes[node]['original_name']}")
    print(f"Latitude: {intersections[node]['latitude']}")
    print(f"Longitude: {intersections[node]['longitude']}")
    print(f"Neighbors: {list(G.neighbors(node))}")


print("Number of nodes:", len(G.nodes))
print("Number of edges:", len(G.edges))
print("Is the graph connected?", nx.is_connected(G))

# Check the distances between road nodes
for edge in G.edges(data=True):
    road1 = edge[0]
    road2 = edge[1]
    distance = edge[2]['weight']
    print(f"Distance between {road1} and {road2}: {distance} km")

