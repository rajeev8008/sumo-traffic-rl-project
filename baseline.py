import os
import sys
import traci # Import the traci library

# --- Configuration ---
SUMO_BINARY = "sumo"  # Use command-line SUMO (no GUI for faster execution)
CONFIG_FILE = "map.sumocfg"
SIM_DURATION = 3600

# --- TraCI Setup ---
print("DEBUG: Checking SUMO_HOME...")
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print(f"DEBUG: Added {tools} to sys.path")
else:
    print("ERROR: SUMO_HOME environment variable not declared.")
    sys.exit("Please declare the 'SUMO_HOME' environment variable.")

# --- Data Collection ---
travel_times = {
    "car": [],
    "bus": [],
    "emergency": []
}
print("DEBUG: Initialized travel_times dictionary.")

# Keep track of when each vehicle departed and its type so we can compute
# travel time on arrival without calling vehicle APIs for already-removed
# vehicles (which raises "vehicle not known").
depart_times = {}   # veh_id -> depart step
veh_types = {}      # veh_id -> vehicle class (car/bus/emergency/...)
print("DEBUG: Initialized depart_times and veh_types maps.")
veh_source = {}     # veh_id -> 'traci'|'inferred'|'class'|None
depart_traci_count = 0
depart_inferred_count = 0
depart_class_mapped = 0
arrival_without_depart = 0

# --- Main Simulation Loop ---
print("Starting SUMO simulation for baseline measurement...")

# Start SUMO with a fixed seed for reproducibility
# Use seed 42 to get consistent baseline results across multiple runs
try:
    print("DEBUG: Attempting traci.start with fixed seed for baseline comparison...")
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--seed=42", "--no-warnings=true"])
    print("DEBUG: traci.start successful.")
except Exception as e_start:
    print(f"ERROR: Failed to start SUMO/Traci: {e_start}")
    sys.exit("Exiting due to Traci start error.")

# Run the simulation steps
step = 0
try:
    while step < SIM_DURATION:
        traci.simulationStep()

        # First, record newly departed vehicles so we capture their type and
        # the time they entered the simulation. At departure time the vehicle
        # still exists and vehicle API calls succeed.
        departed = traci.simulation.getDepartedIDList()
        for veh_id in departed:
            type_id = None
            source = None
            try:
                type_id = traci.vehicle.getTypeID(veh_id)
                source = 'traci'
            except Exception:
                type_id = None
                source = None

            # If we couldn't get a type via TraCI, try to infer it from the
            # vehicle id (flows often produce ids like "car_NS.0", "bus_SN.0").
            if type_id is None:
                try:
                    base = veh_id.split('.')[0]  # e.g. "car_NS"
                    if base.startswith("car") or "car" in base:
                        type_id = "car"
                        source = 'inferred'
                    elif base.startswith("bus") or "bus" in base:
                        type_id = "bus"
                        source = 'inferred'
                    elif base.startswith("emergency") or "emergency" in base:
                        type_id = "emergency"
                        source = 'inferred'
                    else:
                        try:
                            vclass = traci.vehicle.getVehicleClass(veh_id)
                            if vclass == 'passenger':
                                type_id = 'car'
                                source = 'class'
                            elif vclass == 'emergency':
                                type_id = 'emergency'
                                source = 'class'
                            else:
                                type_id = None
                                source = None
                        except Exception:
                            type_id = None
                            source = None
                except Exception:
                    type_id = None
                    source = None

            depart_times[veh_id] = step
            if type_id is not None:
                veh_types[veh_id] = type_id
                veh_source[veh_id] = source
                if source == 'traci':
                    depart_traci_count += 1
                elif source == 'inferred':
                    depart_inferred_count += 1
                elif source == 'class':
                    depart_class_mapped += 1

        # Then process arrivals. The vehicle object is often already removed
        # from the simulation when it appears in the arrived list, so calling
        # traci.vehicle.* on it will raise "vehicle not known". Instead we
        # compute duration from the stored depart time and recorded type.
        arrived_vehicle_ids = traci.simulation.getArrivedIDList()

        for veh_id in arrived_vehicle_ids:
            try:
                vtype = veh_types.get(veh_id)
                dep = depart_times.get(veh_id)

                # Debug info
                print(f"DEBUG: Vehicle {veh_id} arrived. stored_type={vtype!r}, depart_step={dep}")

                if vtype is None and dep is not None:
                    # Try to infer missing type from vehicle id as a last resort
                    try:
                        base = veh_id.split('.')[0]
                        if base.startswith("car") or "car" in base:
                            vtype = 'car'
                        elif base.startswith("bus") or "bus" in base:
                            vtype = 'bus'
                        elif base.startswith("emergency") or "emergency" in base:
                            vtype = 'emergency'
                        if vtype is not None:
                            print(f"DEBUG: Inferred type '{vtype}' for {veh_id} on arrival.")
                    except Exception:
                        vtype = None

                if vtype in travel_times and dep is not None:
                    duration = step - dep
                    travel_times[vtype].append(duration)
                else:
                    if dep is None:
                        arrival_without_depart += 1

                # Clean up maps to avoid growth
                if veh_id in depart_times:
                    del depart_times[veh_id]
                if veh_id in veh_types:
                    del veh_types[veh_id]

            except Exception as e_inner:
                print(f"WARNING: Unexpected error processing arrived vehicle {veh_id}: {e_inner}")

        step += 1
    print("DEBUG: Simulation loop finished normally.") # DEBUG

except Exception as e_main:
    print(f"ERROR: An error occurred during the simulation loop: {e_main}")
    # Still attempt to close traci if the loop failed
    try:
        print("DEBUG: Attempting traci.close() after loop error...")
        traci.close()
        print("DEBUG: traci.close() successful after loop error.")
    except Exception as e_close_err:
        print(f"ERROR: Failed to close traci after loop error: {e_close_err}")
    sys.exit("Exiting due to simulation loop error.")


# --- Simulation End (Normal Path) ---
# This code runs ONLY if the simulation loop completed without error
try:
    print("DEBUG: Attempting traci.close() after normal loop finish...")
    traci.close()
    print("DEBUG: traci.close() successful after normal loop finish.")
except Exception as e_close_normal:
     print(f"ERROR: Failed to close traci after normal loop finish: {e_close_normal}")

print("Simulation finished.") # Original message
print("DEBUG: Proceeding to metric calculation...") # DEBUG

# --- Calculate and Print Metrics ---
print("\n--- Baseline Performance Metrics ---")

try:
    # Ensure travel_times is accessible
    if "travel_times" not in locals() and "travel_times" not in globals():
        print("CRITICAL ERROR: travel_times dictionary is not defined!")
        sys.exit("Cannot calculate metrics.")

    # Calculate Average Car Travel Time
    num_cars = len(travel_times["car"])
    if num_cars > 0:
        avg_car_time = sum(travel_times["car"]) / num_cars
        print(f"Average Car Travel Time:   {avg_car_time:.2f} seconds ({num_cars} finished)")
    else:
        print("No cars finished their routes.")

    # Calculate Average Bus Travel Time
    num_buses = len(travel_times["bus"])
    if num_buses > 0:
        avg_bus_time = sum(travel_times["bus"]) / num_buses
        print(f"Average Bus Travel Time:   {avg_bus_time:.2f} seconds ({num_buses} finished)")
    else:
        print("No buses finished their routes.")

    # Calculate Average Emergency Transit Time
    num_emergency = len(travel_times["emergency"])
    if num_emergency > 0:
        avg_emergency_time = sum(travel_times["emergency"]) / num_emergency
        print(f"Average Emergency Transit Time: {avg_emergency_time:.2f} seconds ({num_emergency} finished)")
    else:
        print("No emergency vehicles finished their routes.")

    # Diagnostics summary
    print("\nBaseline diagnostics:")
    try:
        print(f"  Depart types obtained from Traci: {depart_traci_count}")
        print(f"  Depart types inferred from id: {depart_inferred_count}")
        print(f"  Depart types mapped from vehicle class: {depart_class_mapped}")
        print(f"  Arrivals without recorded depart time: {arrival_without_depart}")
    except Exception:
        pass

except NameError as ne:
    print(f"\nCRITICAL NameError during metric calculation: {ne}") # Specific NameError
except Exception as e_metrics:
    print(f"\nERROR: An unexpected error occurred during metric calculation: {e_metrics}")

print("\nBaseline script finished successfully.")