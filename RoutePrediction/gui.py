import tkinter as tk
from tkinter import ttk
import networkx as nx
import pandas as pd
from process import calculate_time, haversine, find_route, find_closest_nodes_by_scats, visualize_route_on_map, G as G
import webbrowser

# Load the SCATS data into a DataFrame
scats_data = pd.read_excel('Scats Data October 2006.xls', sheet_name='Data', skiprows=1)

def open_map_in_browser():
    webbrowser.open("route_map.html", new=2)

# Define a function to find the intersection based on the entered SCATS number
def find_intersection_by_scats(scats_number, scats_data):
    for index, row in scats_data.iterrows():
        data_scats_number = row['SCATS Number']
        if data_scats_number == scats_number:
            location = row['Location']
            road_names = location.split(" of ")
            intersection_name = " of ".join(road_names)
            print(f"Found intersection for SCATS Number {scats_number}: {intersection_name}")
            return intersection_name

    print(f"SCATS Number {scats_number} not found.")
    return None

def calculate_route():
    # Get the entered SCATS numbers as integers
    start_scats = int(start_scats_var.get())
    end_scats = int(end_scats_var.get())

    print(f"Start SCATS: {start_scats}")
    print(f"End SCATS: {end_scats}")

    start_intersection = find_intersection_by_scats(start_scats, scats_data)
    end_intersection = find_intersection_by_scats(end_scats, scats_data)

    print(f"Start intersection 1: {start_intersection}")
    print(f"End intersection 1: {end_intersection}")

    # Call the function to find the closest start and end nodes based on SCATS numbers
    start_intersection, end_intersection = find_closest_nodes_by_scats(start_scats, end_scats, scats_data, G)

    print(f"Start intersection 2: {start_intersection}")
    print(f"End intersection 2: {end_intersection}")

    if start_intersection and end_intersection:
        route_info = find_route(start_intersection, end_intersection)
        
        print(f"{route_info}")

        if route_info:
            result_label.config(text=f"Route from {start_scats} to {end_scats}:\n"
                                     f"Distance: {route_info['shortest_distance']:.2f} km\n"
                                     f"Total Travel Time: {route_info['total_time_minutes']} minutes {route_info['total_time_seconds']} seconds\n"
                                     f"Shortest Path: {' -> '.join(route_info['shortest_path'])}")
            # Call the function to visualize the route on a map
            visualize_route_on_map(start_intersection, end_intersection, route_info['shortest_path'], G)
        else:
            result_label.config(text="Start or end intersection not found in the graph.")
    else:
        result_label.config(text="Please enter valid SCATS numbers for both start and end intersections.")

root = tk.Tk()
root.title("Route Finder")

frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Label(frm, text="Enter Start SCATS Number:").grid(column=0, row=0)

start_scats_var = tk.StringVar()
start_scats_entry = ttk.Entry(frm, textvariable=start_scats_var)
start_scats_entry.grid(column=1, row=0)

ttk.Label(frm, text="Enter End SCATS Number:").grid(column=0, row=1)

end_scats_var = tk.StringVar()
end_scats_entry = ttk.Entry(frm, textvariable=end_scats_var)
end_scats_entry.grid(column=1, row=1)

calculate_button = ttk.Button(frm, text="Calculate Route", command=calculate_route)
calculate_button.grid(column=0, row=2, columnspan=2)

result_label = ttk.Label(frm, text="")
result_label.grid(column=0, row=3, columnspan=2)

# Create a button to open the map in a browser
open_map_button = ttk.Button(frm, text="Open Map in Browser", command=open_map_in_browser)
open_map_button.grid(column=0, row=4, columnspan=2)

root.mainloop()
