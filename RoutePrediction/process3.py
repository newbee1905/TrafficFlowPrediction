import pandas as pd
import networkx as nx
from math import atan2, radians, sin, cos, sqrt

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

# Create a graph
G = nx.Graph()

# Create a dictionary to store location data
locations = {}

# Iterate through the data and extract location information
for index, row in scats_data.iterrows():
    scats_number = row['SCATS Number']
    location = row['Location']
    latitude = row['NB_LATITUDE']
    longitude = row['NB_LONGITUDE']

    # Add the location to the graph as a node
    G.add_node(location, latitude=latitude, longitude=longitude)

    # Use the location as a key to store latitude and longitude
    if location not in locations:
        locations[location] = {'latitude': latitude, 'longitude': longitude, 'scats_numbers': set()}  # Use a set to store unique SCATS numbers
    # Add the SCATS Number to the set
    locations[location]['scats_numbers'].add(scats_number)

# Define a threshold distance for considering intersections as part of the same road segment
threshold_distance = 0.1  # Adjust this value as needed

# Create a dictionary to store road segments
road_segments = {}

# Iterate through the locations to create edges (road segments)
for location1, data1 in locations.items():
    lat1, lon1 = data1['latitude'], data1['longitude']

    # Find intersecting locations
    for location2, data2 in locations.items():
        if location1 != location2:
            lat2, lon2 = data2['latitude'], data2['longitude']
            distance = calculate_distance(lat1, lon1, lat2, lon2)

            if distance < threshold_distance:
                # Create an edge between location1 and location2
                G.add_edge(location1, location2)

# Now you have a graph G with intersections as nodes and road segments as edges

# Check how many intersections each road segment connects to
for road_segment in G.edges():
    start_location, end_location = road_segment
    neighbors = list(G.neighbors(start_location)) + list(G.neighbors(end_location))
    print(f"Road Segment: {road_segment}")
    print(f"Intersections Connected: {neighbors}")
    print()

# Check how many road segments each intersection connects to
for intersection in G.nodes():
    connected_road_segments = list(G.edges(intersection))
    print(f"Intersection: {intersection}")
    print(f"Connected Road Segments: {connected_road_segments}")
    print()

# Print the total number of nodes and edges in the graph
print(f"Number of nodes in the graph (intersections): {G.number_of_nodes()}")
print(f"Number of edges in the graph (road segments): {G.number_of_edges()}")
