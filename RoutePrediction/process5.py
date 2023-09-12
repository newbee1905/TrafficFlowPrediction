import pandas as pd # For reading excel data
import networkx as nx # For building the graph
from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance
import folium as fm # For map visualisation 

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

# Create a dictionary to store road names
road_nodes = {}

# Iterate through the data and extract intersection information
for index, row in scats_data.iterrows():
    location = row['Location'].lower()  # Convert to lowercase for case insensitivity

    # Split the location into road names
    road_names = location.split(" of ")

    # Handle the case when there's no delimiter "of"
    if len(road_names) == 1:
        road_names = location.split("of")

    # Extract the unique intersection name
    intersection_name = " of ".join(road_names)

    # Add the intersection as a node
    if intersection_name not in G:
        G.add_node(intersection_name, latitude=row['NB_LATITUDE'], longitude=row['NB_LONGITUDE'])
    
    # Add the road names as nodes and store them in the road_nodes dictionary
    for road_name in road_names:
        road_name = road_name.strip()  # Remove leading/trailing spaces
        if road_name not in road_nodes:
            road_nodes[road_name] = []
        road_nodes[road_name].append(intersection_name)

# Iterate through the data again to add edges
for road_name, intersections in road_nodes.items():
    if len(intersections) > 1:
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                intersection1 = intersections[i]
                intersection2 = intersections[j]
                if not G.has_edge(intersection1, intersection2) and intersection1 != intersection2:  # Exclude self-loop edges
                    lat1, lon1 = G.nodes[intersection1]['latitude'], G.nodes[intersection1]['longitude']
                    lat2, lon2 = G.nodes[intersection2]['latitude'], G.nodes[intersection2]['longitude']
                    distance = haversine(lat1, lon1, lat2, lon2)
                    G.add_edge(intersection1, intersection2, weight=distance)

# Debug information
for node in G.nodes:
    print(f"Node: {node}")
    print(f"Latitude: {G.nodes[node]['latitude']}")
    print(f"Longitude: {G.nodes[node]['longitude']}")
    print(f"Neighbors: {list(G.neighbors(node))}")

print("Number of nodes:", len(G.nodes))
print("Number of edges:", len(G.edges))
print("Is the graph connected?", nx.is_connected(G))

# Find the ideal route between two intersections
start_intersection = "burwood_rd e of glenferrie_rd"  # Replace with the actual intersection name
end_intersection = "rathmines_rd w of burke_rd"    # Replace with the actual intersection name

if start_intersection in G and end_intersection in G:
    shortest_path = nx.shortest_path(G, start_intersection, end_intersection, weight='weight')
    shortest_distance = nx.shortest_path_length(G, start_intersection, end_intersection, weight='weight')
    
    print(f"Shortest path from {start_intersection} to {end_intersection}:")
    for i in range(len(shortest_path) - 1):
        print(f"Step {i + 1}: Go from {shortest_path[i]} to {shortest_path[i + 1]}")

    print(f"Total distance: {shortest_distance} km")
else:
    print("Start or end intersection not found in the graph.")


# Map Visualisation 

# Create a folium map centered at the average latitude and longitude of the start and end intersections
map_center = ((G.nodes[start_intersection]['latitude'] + G.nodes[end_intersection]['latitude']) / 2,
              (G.nodes[start_intersection]['longitude'] + G.nodes[end_intersection]['longitude']) / 2)
m = fm.Map(location=map_center, zoom_start=14)

# Add markers for the start and end intersections
start_marker = fm.Marker(
    location=(G.nodes[start_intersection]['latitude'], G.nodes[start_intersection]['longitude']),
    popup=start_intersection,
    icon=fm.Icon(color='green')
).add_to(m)

end_marker = fm.Marker(
    location=(G.nodes[end_intersection]['latitude'], G.nodes[end_intersection]['longitude']),
    popup=end_intersection,
    icon=fm.Icon(color='red')
).add_to(m)

# Create a list of coordinates for the PolyLine representing the route
route_coordinates = [(G.nodes[node]['latitude'], G.nodes[node]['longitude']) for node in shortest_path]

# Add a PolyLine to the map
route_line = fm.PolyLine(
    locations=route_coordinates,
    color='blue',
    weight=5,
    opacity=0.7
).add_to(m)

# Display the map
m.save('route_map.html')

