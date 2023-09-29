import pandas as pd # For reading excel data
import networkx as nx # For building the graph
from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance
import folium as fm # For map visualisation 
import requests
import re
import json


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

# Function to calculate time to travel route based on distance and traffic flow
def calculate_time_with_flow(distance_km, traffic_flow):
    # Adjusted model for traffic flow effect on speed
    k = 0.00003
    effective_speed = 60.0 / (1 + k * traffic_flow ** 2)
    
    # Calculate time for traveling the distance at the effective speed in seconds
    time_seconds = (distance_km / effective_speed) * 3600.0
    
    # Calculate time for intersections (30 seconds per intersection) in seconds
    intersection_time_seconds = 30.0
    
    # Total time is the sum of travel time and intersection time
    total_time_seconds = time_seconds + intersection_time_seconds
    
    # Convert total time to minutes and seconds
    total_time_minutes = int(total_time_seconds // 60)
    total_time_seconds = int(total_time_seconds % 60)
    
    return total_time_minutes, total_time_seconds


# Function to find route between intersections
def find_route(start_intersection, end_intersection, selected_time, selected_model):
    # Check if start and end intersections are in the graph
    if start_intersection in G and end_intersection in G:
        # Calculate the shortest path and distance FIRST
        shortest_path = nx.shortest_path(G, start_intersection, end_intersection, weight='weight')
        shortest_distance = nx.shortest_path_length(G, start_intersection, end_intersection, weight='weight')
        
        total_time_minutes = 0
        total_time_seconds = 0
        
        # Calculate time for each road segment considering traffic flow
        for i in range(len(shortest_path) - 1):
            intersection1 = shortest_path[i]
            intersection2 = shortest_path[i + 1]
            
            # Get the distance between two intersections
            distance = G[intersection1][intersection2]['distance']  # Note the key change to 'distance'
            
            # Fetch traffic flow prediction from the server
            traffic_flow = get_traffic_flow(selected_time, intersection1, selected_model)
            
            # Calculate the time for this road segment in minutes and seconds considering traffic flow
            segment_time_minutes, segment_time_seconds = calculate_time_with_flow(distance, traffic_flow)
            
            # Update the total time
            total_time_minutes += segment_time_minutes
            total_time_seconds += segment_time_seconds
            
            # Adjust total time if there are more than 59 seconds
            if total_time_seconds >= 60:
                total_time_minutes += total_time_seconds // 60
                total_time_seconds = total_time_seconds % 60
        
        return {
            "start_intersection": start_intersection,
            "end_intersection": end_intersection,
            "shortest_path": shortest_path,
            "shortest_distance": shortest_distance,
            "total_time_minutes": total_time_minutes,
            "total_time_seconds": total_time_seconds,
        }
    else:
        return None

    
# Function to fetch traffic flow prediction from the server
def get_traffic_flow(selected_time, intersection_name, selected_model):
    # Convert the model to lowercase
    selected_model = selected_model.lower()

    # Check if the intersection_name exists in the graph nodes
    if intersection_name in G.nodes:
        # Get the node data which contains latitude and longitude
        node_data = G.nodes[intersection_name]
        lat = node_data['latitude']
        lng = node_data['longitude']

        # Define the JSON payload
        data = {
            "start_time": selected_time,
            "lat": lat,  # Use the latitude from node data
            "lng": lng,  # Use the longitude from node data
            "model": selected_model  # Use the lowercase model
        }

        # Convert the data to JSON format
        json_data = json.dumps(data)

        # Set the URL of your FastAPI server
        url = "http://127.0.0.1:8000"

        try:
            # Make a POST request to the server
            response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

            # Check the response
            if response.status_code == 200:
                flow_prediction = response.json()
                print("Flow Prediction:", flow_prediction)
                return flow_prediction
            else:
                # Handle error gracefully
                print(f"Error: Server responded with status code {response.status_code}. Using default flow prediction of 0.")
                return 0  # Default value

        except requests.RequestException as e:
            # Handle request exceptions such as timeouts, connection errors, etc.
            print(f"Error: {e}. Using default flow prediction of 0.")
            return 0  # Default value

    else:
        # Handle the case where the intersection is not found in the graph
        print(f"Intersection '{intersection_name}' not found in the graph.")
        return 0  # Default value


# Function to calculate edge weight based on distance and traffic flow
def calculate_weight(distance, traffic_flow):
    return distance / max(traffic_flow, 1)  # Avoid division by zero

def find_closest_nodes_by_scats(start_scats, end_scats, scats_data, G):
    start_nodes = []
    end_nodes = []
    min_distance = float('inf')

    # Convert SCATS numbers to lowercase
    start_scats = start_scats
    end_scats = end_scats

    for node in G.nodes:
        print(f"Node: {node}, SCATS Number: {G.nodes[node]['scats_number']}")

    # Find all nodes with the entered start SCATS number
    for node in G.nodes:
        node_scats = G.nodes[node]['scats_number']
        if node_scats == start_scats:
            start_nodes.append(node)

    # Find all nodes with the entered end SCATS number
    for node in G.nodes:
        node_scats = G.nodes[node]['scats_number']
        if node_scats == end_scats:
            end_nodes.append(node)

    # Initialize variables to store the selected start and end nodes
    selected_start_node = None
    selected_end_node = None

    # Iterate over all combinations of start and end nodes
    for start_node in start_nodes:
        for end_node in end_nodes:
            try:
                # Attempt to find a path between the current start and end nodes
                distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')

                # Check if this distance is shorter than the minimum found so far
                if distance < min_distance:
                    min_distance = distance
                    selected_start_node = start_node
                    selected_end_node = end_node
            except nx.NetworkXNoPath:
                # No path found between these nodes, continue to the next combination
                continue

    if selected_start_node is None or selected_end_node is None:
        # Print an error message when no connection is found
        print("Error: No valid path found between start and end nodes.")

    return selected_start_node, selected_end_node


def visualize_route_on_map(start_intersection, end_intersection, shortest_path, G):
    # Function to geocode an intersection name into coordinates
    def geocode_intersection(intersection_name):
        try:
            # Use Nominatim to geocode the intersection name into coordinates
            cleaned_name = preprocess_intersection_name(intersection_name)
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={cleaned_name}"
            response = requests.get(url)
            data = response.json()
            if data:
                # Check if multiple locations were returned
                if len(data) > 1:
                    print(f"Multiple locations found for intersection: {intersection_name}")
                    return None
                location = data[0]
                return float(location['lat']), float(location['lon'])
        except Exception as e:
            print(f"Error geocoding {intersection_name}: {e}")
        return None

    # Function to process intersection names to work with openstreetmap
    def preprocess_intersection_name(name):
        # Use regular expressions to extract road names without "of" and directional indicators
        cleaned_name = re.sub(r'(\w+)\s?[nswe]+[sw]*\s?of\s?(\w+)', r'\1 \2', name, flags=re.IGNORECASE)
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
                # If geocoding fails, use the coordinates from the graph nodes instead as a fallback
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
        G.add_node(intersection_name, latitude=row['NB_LATITUDE'], longitude=row['NB_LONGITUDE'], scats_number=row['SCATS Number'])
    
    
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
                    G.add_edge(intersection1, intersection2, weight=distance, distance=distance)

