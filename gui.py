import tkinter as tk
from tkinter import ttk
import networkx as nx
import pandas as pd
from process import get_traffic_flow, find_route, find_closest_nodes_by_scats, visualize_route_on_map, G as G
import webbrowser

# Load the SCATS data into a DataFrame
scats_data = pd.read_excel('Scats Data October 2006.xls', sheet_name='Data', skiprows=1)

# Time options for traffic volume
time_options = ["0:00", "0:15", "0:30", "0:45", "1:00", "1:15", "1:30", "1:45", "2:00", "2:15", "2:30", "2:45", "3:00", "3:15", "3:30", "3:45", "4:00", "4:15", "4:30", "4:45", "5:00", "5:15", "5:30", "5:45", "6:00", "6:15", "6:30", "6:45", "7:00", "7:15", "7:30", "7:45", "8:00", "8:15", "8:30", "8:45", "9:00", "9:15", "9:30", "9:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45", "16:00", "16:15", "16:30", "16:45", "17:00", "17:15", "17:30", "17:45", "18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45", "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45"]

# Global variable to store the routes
all_routes = []

def open_map_in_browser():
    webbrowser.open("route_map.html", new=2)

def calculate_route():
    global all_routes
    all_routes = []  # Clear previous routes

    # Get the entered SCATS numbers as integers
    start_scats = int(start_scats_var.get())
    end_scats = int(end_scats_var.get())

    # Call the function to find the closest start and end nodes based on SCATS numbers
    start_intersection, end_intersection = find_closest_nodes_by_scats(start_scats, end_scats, scats_data, G)

    # Get the selected ML model
    selected_model = ml_model_var.get()

    # Get the selected time
    selected_time = time_dropdown.get()

    # Get the selected algorithm
    selected_algorithm = algorithm_var.get()

    if start_intersection and end_intersection:
        route_details = find_route(start_intersection, end_intersection, selected_time, selected_model, selected_algorithm)

        if route_details:
            all_routes = route_details  # Store all routes for global access
            update_route_dropdown()  # Update the route dropdown with new values
            display_route(1)  # Display the first route by default

            # Show route selection components
            route_dropdown_label.grid()
            route_dropdown.grid()

            # Show the "Open Map in Browser" button
            open_map_button.grid(column=0, row=8, columnspan=2)
        else:
            result_label.config(text="No routes found from start to end intersection.")
    else:
        result_label.config(text="Please enter valid SCATS numbers for both start and end intersections.")

def display_route(route_number):
    if 1 <= route_number <= len(all_routes):  # Routes start from 1
        route_index = route_number - 1  # Convert to 0-based index
        route_info = all_routes[route_index - 1]  # -1 because list indices are 0-based
        shortest_path = route_info['path']
        formatted_path = "\n".join([f"{i + 1}. {shortest_path[i]} -> {shortest_path[i + 1]}" for i in range(len(shortest_path) - 1)])
        
        result_label.config(text=f"Route {route_index + 1} from {start_scats_var.get()} to {end_scats_var.get()} using {ml_model_var.get()} and {algorithm_var.get()} Search:\n"
                                f"\nDistance: \n{route_info['distance']:.2f} km\n"
                                f"Total Travel Time: \n{route_info['total_time_minutes']} minutes {route_info['total_time_seconds']} seconds\n"
                                f"\nPath:\n{formatted_path}")

        # Call the function to visualize the route on a map
        visualize_route_on_map(route_info['start_intersection'], route_info['end_intersection'], route_info['path'], G)
    else:
        result_label.config(text="Invalid route selection.")

def update_route_dropdown():
    """Update the route dropdown values based on the all_routes global variable."""
    route_options = [i + 1 for i in range(len(all_routes))]  # +1 because routes start from 1
    route_dropdown['values'] = route_options
    if route_options:
        route_dropdown.set(route_options[0])  # Default value is the first route
    else:
        route_dropdown.set("")

def get_intersection_by_scats(scats_number, scats_data, G):
    # Use the existing function to find the closest nodes
    closest_node, _ = find_closest_nodes_by_scats(scats_number, scats_number, scats_data, G)
    # The closest_node is the intersection name you're looking for
    return closest_node

def get_flow_data():
    # Get the entered SCATS number as an integer
    try:
        scats_number = int(scats_var.get())
    except ValueError:
        flow_data_label.config(text="Please enter a valid integer for SCATS number.")
        return
        # Get the intersection name by the SCATS number
    intersection_name = get_intersection_by_scats(scats_number, scats_data, G)

    if intersection_name is None:
        flow_data_label.config(text="No intersection found for the provided SCATS number.")
        return

    # Get the selected time and model
    selected_time = flow_time_dropdown.get()
    selected_model = flow_model_dropdown.get()

    # Call the get_traffic_flow method
    flow_data = get_traffic_flow(selected_time, intersection_name, selected_model)

    # Display the flow data in the GUI
    flow_data_label.config(text=f"Traffic Volume for SCATS site: {scats_number}\n{intersection_name}\nAt {selected_time} using {selected_model}:\n{flow_data}")


root = tk.Tk()
root.title("Route Finder")

frm = ttk.Frame(root, padding=10)
frm.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
frm.columnconfigure(0, minsize=20)  
frm.columnconfigure(1, minsize=20)  



# Route prediction

ttk.Label(frm, text="Enter Start SCATS Number:").grid(column=0, row=0, sticky=tk.W)

start_scats_var = tk.StringVar()
start_scats_entry = ttk.Entry(frm, textvariable=start_scats_var, width=20)
start_scats_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

ttk.Label(frm, text="Enter End SCATS Number:").grid(column=0, row=1, sticky=tk.W)

end_scats_var = tk.StringVar()
end_scats_entry = ttk.Entry(frm, textvariable=end_scats_var, width=20)
end_scats_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

# Dropdown menu for time selection
ttk.Label(frm, text="Select Start Time:").grid(column=0, row=2, sticky=tk.W)
time_dropdown = ttk.Combobox(frm, values=time_options, width=20) 
time_dropdown.grid(column=1, row=2, sticky=tk.W)
time_dropdown.set("0:00") 

# Dropdown menu for ML model selection
ttk.Label(frm, text="Select ML Model:").grid(column=0, row=3, sticky=tk.W)
ml_model_options = ["LSTM", "GRU", "SAES", "CNN", "Prophet"]
ml_model_var = tk.StringVar()
ml_model_dropdown = ttk.Combobox(frm, values=ml_model_options, textvariable=ml_model_var, width=20) 
ml_model_dropdown.grid(column=1, row=3, sticky=tk.W)
ml_model_dropdown.set("LSTM") 

# Dropdown menu for algorithm selection
ttk.Label(frm, text="Select Search Algorithm:").grid(column=0, row=4, sticky=tk.W)  
algorithm_options = ["Dijkstra", "A*", "Bi-Directional", "Yen's K-Shortest"]
algorithm_var = tk.StringVar()
algorithm_dropdown = ttk.Combobox(frm, values=algorithm_options, textvariable=algorithm_var, width=20) 
algorithm_dropdown.grid(column=1, row=4, sticky=tk.W) 
algorithm_dropdown.set("Dijkstra")  # Default value

# Button to calculate a route
calculate_button = ttk.Button(frm, text="Calculate Route", command=calculate_route)
calculate_button.grid(column=0, row=5, columnspan=2)


# Create label for route dropdown (initially hidden)
route_dropdown_label = ttk.Label(frm, text="Select Route:")
route_dropdown_label.grid(column=0, row=6, sticky=tk.W)
route_dropdown_label.grid_remove()

# Create route dropdown (initially hidden)
route_var = tk.IntVar()
route_dropdown = ttk.Combobox(frm, textvariable=route_var, width=20, state="readonly") 
route_dropdown.grid(column=1, row=6, sticky=tk.W)
route_dropdown.bind("<<ComboboxSelected>>", lambda e: display_route(route_var.get()))
route_dropdown.grid_remove()

result_label = ttk.Label(frm, text="")
result_label.grid(column=0, row=7, columnspan=2)

# Button to open the map in a browser (initially hidden)
open_map_button = ttk.Button(frm, text="Open Map in Browser", command=open_map_in_browser)
open_map_button.grid(column=0, row=9, columnspan=2)
open_map_button.grid_remove()  # Hide the button initially



# Volume Prediction

# Entry field for scats site for volume prediction
ttk.Label(frm, text="Enter SCATS Number for Traffic Volume:").grid(column=0, row=11, sticky=tk.W)
scats_var = tk.StringVar()
scats_entry = ttk.Entry(frm, textvariable=scats_var, width=20)
scats_entry.grid(column=1, row=11, sticky=(tk.W, tk.E))

# Dropdown for time selection specific to flow data
ttk.Label(frm, text="Select Time for Volume Prediction:").grid(column=0, row=12, sticky=tk.W)
flow_time_dropdown = ttk.Combobox(frm, values=time_options, width=20) 
flow_time_dropdown.grid(column=1, row=12, sticky=tk.W)
flow_time_dropdown.set("0:00") 

# Dropdown for ML model selection specific to flow data
ttk.Label(frm, text="Select Model for Predicting Traffic Volume:").grid(column=0, row=13, sticky=tk.W)
flow_model_options = ["LSTM", "GRU", "SAES", "CNN", "Prophet"]
flow_model_var = tk.StringVar()
flow_model_dropdown = ttk.Combobox(frm, values=flow_model_options, textvariable=flow_model_var, width=20) 
flow_model_dropdown.grid(column=1, row=13, sticky=tk.W)
flow_model_dropdown.set("LSTM") 

# Button to get flow data
get_flow_button = ttk.Button(frm, text="Predict Volume", command=get_flow_data)
get_flow_button.grid(column=0, row=14, columnspan=2)

# Label to display the flow data
flow_data_label = ttk.Label(frm, text="")
flow_data_label.grid(column=0, row=24, columnspan=2, sticky=(tk.W, tk.E))

root.mainloop()






