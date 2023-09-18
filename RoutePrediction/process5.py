import pandas as pd # For reading excel data
import networkx as nx # For building the graph
from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance
import folium as fm # For map visualisation 
import requests
import re

# Function to calculate length of road segments
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

# Function to calculate time to travel route
def calculate_time(distance_km):
    # Calculate time for traveling the distance at 60 km/h in seconds
    time_seconds = (distance_km / 60.0) * 60.0 * 60.0
    
    # Calculate time for intersections (30 seconds per intersection) in seconds
    intersection_time_seconds = 30.0
    
    # Total time is the sum of travel time and intersection time
    total_time_seconds = time_seconds + intersection_time_seconds
    
    # Convert total time to minutes and seconds
    total_time_minutes = int(total_time_seconds // 60)
    total_time_seconds = int(total_time_seconds % 60)
    
    return total_time_minutes, total_time_seconds



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

# Find the ideal route between two intersections (hardcoded example)
start_intersection = "bridge_rd sw of burwood_rd"  # We can modify to be whatever the user selects in the GUI
end_intersection = "offramp_eastern_fwy e of bulleen_rd"    

# Routing using Djikstra search method - can add or adjust as required
if start_intersection in G and end_intersection in G:
    shortest_path = nx.shortest_path(G, start_intersection, end_intersection, weight='weight')
    shortest_distance = nx.shortest_path_length(G, start_intersection, end_intersection, weight='weight')
    
    print(f"Shortest path from {start_intersection} to {end_intersection}:")
    
    total_time_minutes = 0  # Initialize total time in minutes
    total_time_seconds = 0  # Initialize total time in seconds
    
    for i in range(len(shortest_path) - 1):
        intersection1 = shortest_path[i]
        intersection2 = shortest_path[i + 1]
        
        # Get the distance between two intersections
        distance = G[intersection1][intersection2]['weight']
        
        # Calculate the time for this road segment in minutes and seconds
        segment_time_minutes, segment_time_seconds = calculate_time(distance)
        
        # Update the total time
        total_time_minutes += segment_time_minutes
        total_time_seconds += segment_time_seconds
        
        # Adjust total time if there are more than 59 seconds
        if total_time_seconds >= 60:
            total_time_minutes += total_time_seconds // 60
            total_time_seconds = total_time_seconds % 60
        
        print(f"Step {i + 1}: Go from {intersection1} to {intersection2}, Time: {segment_time_minutes} minutes {segment_time_seconds} seconds")
    
    print(f"Total distance: {shortest_distance:.2f} km")
    print(f"Total time: {total_time_minutes} minutes {total_time_seconds} seconds")
else:
    print("Start or end intersection not found in the graph.")


# Map Visualisation 

# Function to geocode an intersection name into coordinates
def geocode_intersection(intersection_name):
    try:
        # Use Nominatim to geocode the intersection name into coordinates
        cleaned_name = preprocess_intersection_name(intersection_name)
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={cleaned_name}"
        response = requests.get(url)
        data = response.json()
        if data:
            location = data[0]
            return float(location['lat']), float(location['lon'])
    except Exception as e:
        print(f"Error geocoding {intersection_name}: {e}")
    return None

# Function to process intersection names to work with openstreetmap
def preprocess_intersection_name(name):
    # Use regular expressions to extract road names without "of" and directional indicators
    cleaned_name = re.sub(r'(\w+)\s?[nsew]\s?of\s?(\w+)', r'\1 \2', name, flags=re.IGNORECASE)
    print(cleaned_name)
    return cleaned_name


# Get coordinates for start and end intersections
start_coordinates = geocode_intersection(start_intersection)
end_coordinates = geocode_intersection(end_intersection)

# Handle start intersection not found
if start_coordinates is None:
    print(f"Geocoding failed for start intersection: {start_intersection}")
    # Check if start intersection coordinates are available in the graph nodes
    if start_intersection in G.nodes:
        node_data = G.nodes[start_intersection]
        start_coordinates = (node_data['latitude'], node_data['longitude'])
        print(f"Using coordinates from graph nodes for start intersection: {start_intersection}")
    else:
        print(f"No coordinates found for start intersection: {start_intersection}")

# Handle end intersection not found
if end_coordinates is None:
    print(f"Geocoding failed for end intersection: {end_intersection}")
    # Check if end intersection coordinates are available in the graph nodes
    if end_intersection in G.nodes:
        node_data = G.nodes[end_intersection]
        end_coordinates = (node_data['latitude'], node_data['longitude'])
        print(f"Using coordinates from graph nodes for end intersection: {end_intersection}")
    else:
        print(f"No coordinates found for end intersection: {end_intersection}")


if start_coordinates and end_coordinates:
    # Create a folium map centered at the average latitude and longitude of the start and end intersections
    map_center = ((start_coordinates[0] + end_coordinates[0]) / 2, (start_coordinates[1] + end_coordinates[1]) / 2)
    m = fm.Map(location=map_center, zoom_start=14)

    # Add markers for the start and end intersections
    start_marker = fm.Marker(
        location=start_coordinates,
        popup=start_intersection,
        icon=fm.Icon(color='green')
    ).add_to(m)

    end_marker = fm.Marker(
        location=end_coordinates,
        popup=end_intersection,
        icon=fm.Icon(color='red')
    ).add_to(m)

    # Create a list to store coordinates for the route, including the start and end intersections
    route_coordinates = [start_coordinates]

# Add markers for the intersections along the route
    for i in range(len(shortest_path)):
        intersection_name = shortest_path[i]
        intersection_coordinates = geocode_intersection(intersection_name)
        if intersection_coordinates:
            # Add the intersection coordinates to the route
            route_coordinates.append(intersection_coordinates)

            # Create a marker for the intersection
            intersection_marker = fm.Marker(
                location=intersection_coordinates,
                popup=intersection_name,
                icon=fm.Icon(color='blue')
            ).add_to(m)
        else:
            print(f"Geocoding failed for intersection: {intersection_name}")
            # If geocoding fails, use the coordinates from the graph nodes instead as fall back
            if intersection_name in G.nodes:
                node_data = G.nodes[intersection_name]
                intersection_coordinates = (node_data['latitude'], node_data['longitude'])
                route_coordinates.append(intersection_coordinates)
                # Create a marker for the intersection
                intersection_marker = fm.Marker(
                    location=intersection_coordinates,
                    popup=intersection_name,
                    icon=fm.Icon(color='blue')
                ).add_to(m)
            else:
                print(f"No coordinates found for intersection: {intersection_name}")

    # Add a PolyLine to visualize the route
    route_line = fm.PolyLine(
        locations=route_coordinates,
        color='blue',
        weight=5,
        opacity=0.7
    ).add_to(m)

    # Display the map
    m.save('route_map.html')
else:
    print("Start or end intersection not found in the geocoding service.")

