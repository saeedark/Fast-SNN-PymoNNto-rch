import os
import sys
import platform
import subprocess
import pandas as pd
from tqdm import tqdm


def get_simulation_time(byte_out):
    str_out = byte_out.decode("utf-8")
    result = [
        line for line in str_out.split("\n") if line.startswith("simulation time:")
    ]
    return result

def get_simulation_spikes_count(byte_out):
    str_out = byte_out.decode("utf-8")
    result = [
        line for line in str_out.split("\n") if line.startswith("Total spikes:")
    ]
    return result

def get_number_time(line, script_loc):
    try:
        time = float(line.lstrip("simulation time:"))
    except RuntimeError:
        print(f"ERROR: couldn't get the time from string on script {script_loc}, line:")
        print(line)
        return None
    return time

def get_number_spikes(line, script_loc):
    try:
        time = float(line.lstrip("Total spikes:"))
    except RuntimeError:
        print(f"ERROR: couldn't get the spikes count from string on script {script_loc}, line:")
        print(line)
        return None
    return time

def get_script_time(script_loc):
    result = get_simulation_time(subprocess.check_output(["python3", script_loc, 'no_plot']))
    if len(result) == 0:
        print(f"ERROR: no simulation time on {script_loc}")
    elif len(result) > 1:
        print(f"ERROR: multiple simulation time on {script_loc}")
    else:
        return get_number_time(result[0], script_loc)
    return None

def run_simulator(script_loc):
    sim_output = subprocess.check_output(["python3", script_loc, 'no_plot'])
    result_time = get_simulation_time(sim_output)
    result_sipkes = get_simulation_spikes_count(sim_output)
    if len(result_time) == 0 or len(result_sipkes) == 0:
        print(f"ERROR: no simulation time or spikes count on {script_loc}")
    elif len(result_time) > 1 or len(result_sipkes) > 1:
        print(f"ERROR: multiple simulation time or spikes count on {script_loc}")
    else:
        return get_number_time(result_time[0], script_loc), get_number_spikes(result_sipkes[0], script_loc)
    return None


if __name__ == "__main__":
    n = int(sys.argv[1])  # repeat
    out_loc = (
        f"{platform.node()}" if sys.argv[2] == "_" else sys.argv[2]
    )  # csv output location
    scripts_loc = sys.argv[3:]  # list of script to run

    result_time = {}
    result_spike = {}

    for _ in tqdm(range(n)):
        for script_loc in tqdm(scripts_loc):
            script_name = os.path.basename(script_loc)
            if script_name not in result_time:
                result_time[script_name] = []
                result_spike[script_name] = []
            t,s = run_simulator(script_loc)
            if t is not None:
                result_time[script_name].append(t)
                result_spike[script_name].append(s)

    new_df_time = pd.DataFrame(result_time)
    df_time = pd.DataFrame({})
    time_out_loc = out_loc + "_time.csv"
    if os.path.exists(time_out_loc):
        df_time = pd.read_csv(time_out_loc, index_col=0)

    result_df_time = pd.concat([df_time, new_df_time], ignore_index=True, sort=False)
    result_df_time.to_csv(time_out_loc)

    new_df_spike = pd.DataFrame(result_spike)
    df_spike = pd.DataFrame({})
    spike_out_loc = out_loc + "_spikes.csv"
    if os.path.exists(spike_out_loc):
        df_spike = pd.read_csv(spike_out_loc, index_col=0)

    result_df_spike = pd.concat([df_spike, new_df_spike], ignore_index=True, sort=False)
    result_df_spike.to_csv(spike_out_loc)
