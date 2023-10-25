import re
import folium as fm # For map visualisation
import requests

def visualize_route_on_map(start_intersection, end_intersection, shortest_path, G, coords_flag):
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

    # Function to retrieve coordinates based on the flag
    def get_coordinates(intersection_name):
        # If coords_flag is 1, use the API to fetch coordinates
        if coords_flag == 1:
            coordinates = geocode_intersection(intersection_name)
            # If geocoding fails, use the default coordinates
            if coordinates is None:
                coordinates = G.nodes[intersection_name].get('latitude', None), G.nodes[intersection_name].get('longitude', None)
        # If coords_flag is 0, use the default coordinates from the graph nodes
        else:
            coordinates = G.nodes[intersection_name].get('latitude', None), G.nodes[intersection_name].get('longitude', None)
        
        return coordinates

    start_coordinates = get_coordinates(start_intersection)
    end_coordinates = get_coordinates(end_intersection)

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
            intersection_coordinates = get_coordinates(intersection_name)
            if intersection_coordinates != (None, None):
                # Add the intersection coordinates to the route
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