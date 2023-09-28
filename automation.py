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


def get_number(line, script_loc):
    try:
        time = float(line.lstrip("simulation time:"))
    except RuntimeError:
        print(f"ERROR: couldn't get the time from string on script {script_loc}, line:")
        print(line)
        return None
    return time


def get_script_time(script_loc, n):
    result = get_simulation_time(subprocess.check_output(["python3", script_loc, str(n), 'no_plot']))
    if len(result) == 0:
        print(f"ERROR: no simulation time on {script_loc}")
    elif len(result) > 1:
        print(f"ERROR: multiple simulation time on {script_loc}")
    else:
        return get_number(result[0], script_loc)
    return None


if __name__ == "__main__":
    n = int(sys.argv[1])  # repeat
    out_loc = (
        f"{platform.node()}.csv" if sys.argv[2] == "_" else sys.argv[2]
    )  # csv output location
    scripts_loc = sys.argv[3:]  # list of script to run

    result = []
    
    max_reached = {script_loc: False for script_loc in scripts_loc}
    
    a = [10, 20, 50, 100, 250, 500, 750, 1000, 1500, 2000]
    b = [1000*x for x in range(3, 16)]
    

    for size in tqdm(a+b):
        for _ in tqdm(range(n)):
            for script_loc in tqdm(scripts_loc):
                if not max_reached[script_loc]:
                    try:
                        script_name = os.path.basename(script_loc)
                        t = get_script_time(script_loc, size)
                        if t is not None:
                            result.append([script_name, size, t])
                    except:
                        max_reached[script_loc] = True
                    
    df = pd.DataFrame({})
    if os.path.exists(out_loc):
        df = pd.read_csv(out_loc, index_col=0)

    result_df = pd.DataFrame(result, columns=['script_name', 'size', 'time'])
    result_df = pd.concat([df, result_df])
    result_df.to_csv(out_loc)
