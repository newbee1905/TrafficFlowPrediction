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

# Function to find the intersection based on the entered SCATS number
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

    start_intersection = find_intersection_by_scats(start_scats, scats_data)
    end_intersection = find_intersection_by_scats(end_scats, scats_data)

    # Call the function to find the closest start and end nodes based on SCATS numbers
    start_intersection, end_intersection = find_closest_nodes_by_scats(start_scats, end_scats, scats_data, G)


    if start_intersection and end_intersection:
        route_info = find_route(start_intersection, end_intersection)
        if route_info:
            shortest_path = route_info['shortest_path']
            formatted_path = "\n".join([f"{i + 1}. {shortest_path[i]} -> {shortest_path[i + 1]}" for i in range(len(shortest_path) - 1)])
            result_label.config(text=f"Route from {start_scats} to {end_scats}:\n"
                                    f"\nDistance: \n{route_info['shortest_distance']:.2f} km\n"
                                    f"Total Travel Time: \n{route_info['total_time_minutes']} minutes {route_info['total_time_seconds']} seconds\n"
                                    f"\nShortest Path:\n{formatted_path}")
            
            # Show the "Open Map in Browser" button
            open_map_button.grid(column=0, row=5, columnspan=2)
            
            # Call the function to visualize the route on a map
            visualize_route_on_map(start_intersection, end_intersection, route_info['shortest_path'], G)
        else:
            result_label.config(text="Start or end intersection not found in the graph.")
    else:
        result_label.config(text="Please enter valid SCATS numbers for both start and end intersections.")

root = tk.Tk()
root.title("Route Finder")

frm = ttk.Frame(root, padding=10)
frm.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

ttk.Label(frm, text="Enter Start SCATS Number:").grid(column=0, row=0, sticky=tk.W)

start_scats_var = tk.StringVar()
start_scats_entry = ttk.Entry(frm, textvariable=start_scats_var)
start_scats_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

ttk.Label(frm, text="Enter End SCATS Number:").grid(column=0, row=1, sticky=tk.W)

end_scats_var = tk.StringVar()
end_scats_entry = ttk.Entry(frm, textvariable=end_scats_var)
end_scats_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

# Dropdown menu for time selection
ttk.Label(frm, text="Select Start Time:").grid(column=0, row=2, sticky=tk.W)
time_options = ["0:00", "0:15", "0:30", "0:45", "1:00", "1:15", "1:30", "1:45", "2:00", "2:15", "2:30", "2:45", "3:00", "3:15", "3:30", "3:45", "4:00", "4:15", "4:30", "4:45", "5:00", "5:15", "5:30", "5:45", "6:00", "6:15", "6:30", "6:45", "7:00", "7:15", "7:30", "7:45", "8:00", "8:15", "8:30", "8:45", "9:00", "9:15", "9:30", "9:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45", "16:00", "16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45", "18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45", "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45"]
time_dropdown = ttk.Combobox(frm, values=time_options, width=20)  # Set the width
time_dropdown.grid(column=1, row=2, sticky=tk.W)
time_dropdown.set("0:00")  # Set

# Button to calculate a route
calculate_button = ttk.Button(frm, text="Calculate Route", command=calculate_route)
calculate_button.grid(column=0, row=3, columnspan=2)

result_label = ttk.Label(frm, text="")
result_label.grid(column=0, row=4, columnspan=2)

# Button to open the map in a browser (initially hidden)
open_map_button = ttk.Button(frm, text="Open Map in Browser", command=open_map_in_browser)
open_map_button.grid(column=0, row=5, columnspan=2)
open_map_button.grid_remove()  # Hide the button initially

root.mainloop()

