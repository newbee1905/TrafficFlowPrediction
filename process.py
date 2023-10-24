import pandas as pd # For reading excel data
import networkx as nx # For building the graph
from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance
import folium as fm # For map visualisation 
import requests
import re
import json
import heapq
import copy

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


# Function to add traffic flow consideration to edge weight
def adjust_edge_weight_with_traffic(G, edge, traffic_flow):
    distance = G.edges[edge]['distance']

    # Calculate expected travel time considering traffic flow and distance
    time_minutes, time_seconds = calculate_time_with_flow(distance, traffic_flow)

    # Return the weight of the edge with the calculated travel time (in minutes for simplicity)
    return time_minutes + time_seconds / 60.0


# Dijkstra search to find optimal route
def dijkstra_search(G, start, end, selected_time, selected_model):
    pq = [(0, start, None)]  # Priority queue
    distances = {node: float('inf') for node in G.nodes} # Distance meaning Weight not KM
    distances[start] = 0
    predecessor = {node: None for node in G.nodes}
    # total_time = 0  # Variable to store total travel time

    while pq:
        current_distance, current_node, pred_node = heapq.heappop(pq)

        # Only fetch traffic flow and adjust weights when a node is dequeued, not before (To avoid getting traffic flow for every node)
        if pred_node is not None:
            # Get the traffic flow from the server/API
            traffic_flow = get_traffic_flow(selected_time, current_node, selected_model)

            # Adjust edge weight with the traffic flow
            edge = (pred_node, current_node)
            adjusted_weight = adjust_edge_weight_with_traffic(G, edge, traffic_flow)
            # total_time += adjusted_weight  # Accumulate the total travel time

            # Update distance with actual weight considering traffic flow
            if current_distance + adjusted_weight < distances[current_node]:
                distances[current_node] = current_distance + adjusted_weight
                predecessor[current_node] = pred_node

        if current_node == end:
            break

        for neighbor in G.neighbors(current_node):
            edge = (current_node, neighbor)

            # Only add the base distance, delay the traffic flow fetch (distance meaning weight not physical distance)
            distance = current_distance + G[current_node][neighbor]['weight']

            if distance < distances[neighbor]:
                distances[neighbor] = distance 
                predecessor[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor, current_node))  # Push with predecessor

    # Reconstruct the shortest path from start to end
    path = []
    at = end
    while at is not None:
        path.append(at)
        at = predecessor[at]
    path.reverse()

    return path, distances[end]  # Return the shortest path and the distance


# Straight line Distance Heuristic for A* Search
def heuristic(node1, node2):
    # Get the latitude and longitude for both nodes
    lat1, lon1 = G.nodes[node1]['latitude'], G.nodes[node1]['longitude']
    lat2, lon2 = G.nodes[node2]['latitude'], G.nodes[node2]['longitude']
    
    # Calculate and return the straight-line distance between the two nodes
    return haversine(lat1, lon1, lat2, lon2)

# A* Search
def a_star_search(G, start, end, selected_time, selected_model):
    pq = [(0, start, None)]  # Priority queue
    g_values = {node: float('inf') for node in G.nodes}  # Distance from start to node
    g_values[start] = 0
    f_values = {node: float('inf') for node in G.nodes}  # g + heuristic
    f_values[start] = heuristic(start, end)
    predecessor = {node: None for node in G.nodes}

    while pq:
        current_f_value, current_node, pred_node = heapq.heappop(pq)

        if current_node == end:
            break

        for neighbor in G.neighbors(current_node):
            tentative_g_value = g_values[current_node] + G[current_node][neighbor]['weight']

            if tentative_g_value < g_values[neighbor]:
                predecessor[neighbor] = current_node
                g_values[neighbor] = tentative_g_value
                f_value = tentative_g_value + heuristic(neighbor, end)
                f_values[neighbor] = f_value
                heapq.heappush(pq, (f_value, neighbor, current_node))  # Push with predecessor

                # Only fetch traffic flow and adjust weights when a node is dequeued, not before
                if pred_node is not None:
                    # Get the traffic flow from the server/API
                    traffic_flow = get_traffic_flow(selected_time, neighbor, selected_model)
                    
                    # Adjust edge weight with the traffic flow
                    edge = (pred_node, current_node)
                    adjusted_weight = adjust_edge_weight_with_traffic(G, edge, traffic_flow)

                    # Update g_value with actual weight considering traffic flow
                    if g_values[current_node] + adjusted_weight < g_values[neighbor]:
                        g_values[neighbor] = g_values[current_node] + adjusted_weight
                        f_values[neighbor] = g_values[neighbor] + heuristic(neighbor, end)
                        predecessor[neighbor] = current_node
                        # Update the priority queue
                        heapq.heappush(pq, (f_values[neighbor], neighbor, current_node)) 

    # Reconstruct the shortest path from start to end
    path = []
    at = end
    while at is not None:
        path.append(at)
        at = predecessor[at]
    path.reverse()

    return path, g_values[end]  # Return the shortest path and the distance


# Bi-Directional Search Algorithm
def bidirectional_search(G, start, end, selected_time, selected_model):
    if start == end:
        return [start], 0  # The start and end are the same node

    # Initialize the two searches
    forward_search = {start: (0, None)}  # node: (distance, predecessor)
    backward_search = {end: (0, None)}  # node: (distance, predecessor)
    forward_frontier = [(0, start)]  # priority queue for the forward search
    backward_frontier = [(0, end)]  # priority queue for the backward search
    intersect_node = None  # to keep track of where the searches meet

    while forward_frontier and backward_frontier:
        # Choose the frontier with the smallest priority queue to expand
        if len(forward_frontier) <= len(backward_frontier):
            current_frontier = forward_frontier
            current_search = forward_search
            other_search = backward_search
            direction = 'forward'
        else:
            current_frontier = backward_frontier
            current_search = backward_search
            other_search = forward_search
            direction = 'backward'

        # Pop the node with the shortest distance from the current frontier
        current_distance, current_node = heapq.heappop(current_frontier)

        if current_node in other_search:
            # The searches have met
            intersect_node = current_node
            # Calculate the total distance at this point
            total_distance = current_search[current_node][0] + other_search[current_node][0]
            break

        for neighbor in G.neighbors(current_node):
            # Adjust the weight for the edge
            traffic_flow = get_traffic_flow(selected_time, neighbor, selected_model) 
            edge = (current_node, neighbor) if direction == 'forward' else (neighbor, current_node)
            adjusted_weight = adjust_edge_weight_with_traffic(G, edge, traffic_flow)

            distance = current_distance + adjusted_weight

            if neighbor not in current_search or current_search[neighbor][0] > distance:
                current_search[neighbor] = (distance, current_node)
                heapq.heappush(current_frontier, (distance, neighbor))

    if intersect_node is None:
        return None, float('inf')  # The searches did not meet, so no path exists

    # Reconstruct the path from start to the intersect node
    path = []
    at = intersect_node
    while at is not None:
        path.append(at)
        at = forward_search[at][1]
    path.reverse()

    # Extend the path from the intersect node to the end (excluding the intersect node as it's already added)
    at = backward_search[intersect_node][1]
    while at is not None:
        path.append(at)
        at = backward_search[at][1]

    return path, total_distance  # Return the full path, total distance


# Get up to 5 Routes with Dijkstra search
def yens_k_shortest_paths(G, start, end, selected_time, selected_model, k=5):
    # Determine the shortest path from the start to the end.
    shortest_path, _ = dijkstra_search(G, start, end, selected_time, selected_model)
    if not shortest_path:
        return []  # No path found

    A = [shortest_path]  # List to store the k-shortest paths
    B = []  # Temporary storage for candidate paths
    visited_scats_sites = set()  # Set to store visited SCATS sites

    # Extract SCATS sites from the first path
    for node in shortest_path:
        if 'scats_number' in G.nodes[node]:
            visited_scats_sites.add(G.nodes[node]['scats_number'])

    for i in range(1, k):
        for j in range(len(A[-1]) - 1):
            # Spur node is retrieved from the previous k-shortest path, and the root path is the subpath of the shortest path.
            spur_node = A[-1][j]
            root_path = A[-1][:j + 1]

            # Remove the links that are part of the previous shortest paths which share the same root path.
            edges_removed = []
            for path in A:
                if root_path == path[:j + 1] and (path[j], path[j + 1]) in G.edges:
                    edge_data = G[path[j]][path[j + 1]]
                    G.remove_edge(path[j], path[j + 1])
                    edges_removed.append((path[j], path[j + 1], edge_data))

            # Calculate the spur path from the spur node to the sink.
            spur_path, _ = dijkstra_search(G, spur_node, end, selected_time, selected_model)

            # Entire path is made up of the root path and spur path.
            if spur_path:
                total_path = root_path[:-1] + spur_path

                new_scats_sites = set()
                for node in total_path:
                    if 'scats_number' in G.nodes[node]:
                        new_scats_sites.add(G.nodes[node]['scats_number'])

                if not new_scats_sites.issubset(visited_scats_sites) and total_path not in B:  # Ensure the path has new SCATS sites
                    heapq.heappush(B, (path_length(G, total_path), total_path))
                    visited_scats_sites.update(new_scats_sites)  # Add these SCATS sites to the visited set

            # Add back the edges that were removed from the graph.
            for edge in edges_removed:
                node1, node2, edge_data = edge
                G.add_edge(node1, node2, **edge_data)

        if B:
            # Add the shortest path among the candidates to the k-shortest paths.
            _, path_k = heapq.heappop(B)
            A.append(path_k)
        else:
            break

    return A  # Return the list of k-shortest paths



def path_length(G, path):
    """Calculate the total length of a path."""
    length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        length += G[u][v]['weight']
    return length



# Main function to find route between intersections
def find_route(start_intersection, end_intersection, selected_time, selected_model, algorithm_type):
    # Check if start and end intersections are in the graph
    if start_intersection in G and end_intersection in G:
        routes = []
        if algorithm_type == "Dijkstra":
            shortest_path, shortest_distance = dijkstra_search(G, start_intersection, end_intersection, selected_time, selected_model)
            routes.append((shortest_path, shortest_distance))
        elif algorithm_type == "A*":
            shortest_path, shortest_distance = a_star_search(G, start_intersection, end_intersection, selected_time, selected_model)
            routes.append((shortest_path, shortest_distance))
        elif algorithm_type == "Bi-Directional":
            shortest_path, shortest_distance = bidirectional_search(G, start_intersection, end_intersection, selected_time, selected_model)
            routes.append((shortest_path, shortest_distance))
        elif algorithm_type == "Yen's K-Shortest":
            # Yen's K-Shortest Paths returns multiple paths
            k_paths = yens_k_shortest_paths(G, start_intersection, end_intersection, selected_time, selected_model, k=5)
            for path in k_paths:
                distance = path_length(G, path)  # Calculate the total distance of the path
                routes.append((path, distance))
        else:
            return None  # Error

        # Process each route
        route_details = []
        for path, distance in routes:
            total_time_minutes = 0
            total_time_seconds = 0

            # Calculate time for each road segment considering traffic flow
            for i in range(len(path) - 1):
                intersection1 = path[i]
                intersection2 = path[i + 1]

                # Get the distance between two intersections
                segment_distance = G[intersection1][intersection2]['distance'] 

                # Fetch traffic flow prediction from the server
                traffic_flow = get_traffic_flow(selected_time, intersection1, selected_model)

                # Calculate the time for this road segment in minutes and seconds considering traffic flow
                segment_time_minutes, segment_time_seconds = calculate_time_with_flow(segment_distance, traffic_flow)

                # Update the total time
                total_time_minutes += segment_time_minutes
                total_time_seconds += segment_time_seconds

                # Adjust total time if there are more than 59 seconds
                if total_time_seconds >= 60:
                    total_time_minutes += total_time_seconds // 60
                    total_time_seconds = total_time_seconds % 60

            route_info = {
                "start_intersection": start_intersection,
                "end_intersection": end_intersection,
                "path": path,
                "distance": distance,
                "total_time_minutes": total_time_minutes,
                "total_time_seconds": total_time_seconds,
            }
            route_details.append(route_info)

        return route_details
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

        # URL of FastAPI server
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
                # Default flow of 0 in case of error (Disregard flow)
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


# Define a function to find the closest intersection on the same road
def closest_intersection(current, others, G):
    min_distance = float('inf')
    closest = None
    for other in others:
        if other != current:
            lat1, lon1 = G.nodes[current]['latitude'], G.nodes[current]['longitude']
            lat2, lon2 = G.nodes[other]['latitude'], G.nodes[other]['longitude']
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance < min_distance:
                min_distance = distance
                closest = other
    return closest, min_distance


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
            current_intersection = intersections[i]
            other_intersections = intersections[:i] + intersections[i+1:]
            closest, min_distance = closest_intersection(current_intersection, other_intersections, G)
            if closest and not G.has_edge(current_intersection, closest):  # Exclude self-loop edges
                G.add_edge(current_intersection, closest, weight=min_distance, distance=min_distance)


