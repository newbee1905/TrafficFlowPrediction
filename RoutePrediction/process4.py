import pandas as pd
from math import atan2, radians, sin, cos, sqrt
import networkx as nx

# Function to calculate the distance between two sets of latitude and longitude
def calculate_distance(lat1, lon1, lat2, lon2):
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

# Create a dictionary to store location data
locations = {}

# Iterate through the data and extract location information
for index, row in scats_data.iterrows():
    scats_number = row['SCATS Number']
    location = row['Location']
    latitude = row['NB_LATITUDE']
    longitude = row['NB_LONGITUDE']
    
    # Use the location as a key to store latitude and longitude
    if location not in locations:
        locations[location] = {'latitude': latitude, 'longitude': longitude, 'scats_numbers': set()}  # Use a set to store unique SCATS numbers
    # Add the SCATS Number to the set
    locations[location]['scats_numbers'].add(scats_number)

# Create a dictionary to store road segments
road_segments = {}

# Define a threshold distance for considering intersections as part of the same road segment
threshold_distance = 0.1  # Adjust this value as needed

# Iterate through the locations
for location1, data1 in locations.items():
    lat1, lon1 = data1['latitude'], data1['longitude']

    # Initialize a list to store intersecting locations for the current location
    intersecting_locations = []

    # Find intersecting locations
    for location2, data2 in locations.items():
        if location1 != location2:
            lat2, lon2 = data2['latitude'], data2['longitude']
            distance = calculate_distance(lat1, lon1, lat2, lon2)

            if distance < threshold_distance:
                intersecting_locations.append(location2)

    # Create road segments
    for intersecting_location in intersecting_locations:
        road_name = location1.split('_')[0]  # Extract road name from location
        road_segment_name = f"{road_name} {location1} - {intersecting_location}"
        road_segments[road_segment_name] = {
            'start_location': location1,
            'end_location': intersecting_location,
        }

# Create a graph
G = nx.Graph()

# Iterate through the locations and add them as nodes to the graph
for location, data in locations.items():
    G.add_node(location, latitude=data['latitude'], longitude=data['longitude'])

# Iterate through the road segments and add them as edges to the graph
for road_segment, data in road_segments.items():
    G.add_edge(data['start_location'], data['end_location'])

# Calculate the average number of edges connected to each node
avg_edges_per_node = sum([len(G.edges(node)) for node in G.nodes]) / len(G.nodes)

# Calculate the average number of nodes connected to each edge
avg_nodes_per_edge = sum([len(G[node1][node2]) for node1, node2 in G.edges]) / len(G.edges)

# Print the number of nodes and edges
print(f"Number of nodes in the graph (intersections): {len(G.nodes)}")
print(f"Number of edges in the graph (road segments): {len(G.edges)}")

# Print the average number of edges connected to each node and average number of nodes connected to each edge
print(f"Average number of edges connected to each node: {avg_edges_per_node}")
print(f"Average number of nodes connected to each edge: {avg_nodes_per_edge}")

