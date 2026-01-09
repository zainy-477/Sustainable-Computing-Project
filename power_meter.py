# power_meter.py - extracts a running total of power consumption from JSON data
# Zain Hirji, 14th December 2025


# 1. Imports
import requests
import re
import time
from power_metrics import walk
import matplotlib.pyplot as plt
import threading
import numpy as np
import statsmodels.api as sm
from regression_method import regression
from recomposition_method import recomposition


# 2. Set desired locations and variables
power_url = "http://localhost:8085/data.json"
num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
carbon_intensity_api = "https://api.carbonintensity.org.uk/regional/regionid/13"


# 3. Obtain node from path function
def get_node(root, path):
    node = root

    # 3.1 Iterate through each segment in the path to find the corresponding node
    for seg in (s for s in path.split("/") if s):
        for child in node.get("Children", []) or []:
            if child.get("Text") == seg:
                node = child
                break

    # 3.2 Return the dictionary of the found node
    return node
    

# 4. Poll class to repeatedly fetch JSON data and extract total power consumption
class PowerPoller:

    # 4.1 Initialisation
    def __init__(self, url, target_paths, api, interval=0.25):
        self.url = url
        self.api = api
        self.target_paths = target_paths
        self.interval = interval

        self.running = False
        self.thread = None

        self.data = []
        self.carbon_intensity = 0
        self.avg_power = {}
        self.duration = 0

    # 4.2 Start method for polling
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.poll, daemon = True)
            self.thread.start()

    # 4.3 Stop method for polling
    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()

    # 4.4 Poll method to fetch data at regular intervals
    def poll(self):
        
        # 4.4.1 Fetch current carbon intensity data in gCO2/kWh
        response = requests.get(self.api, timeout=2)
        data = response.json()
        self.carbon_intensity = data['data'][0]['data'][0]['intensity']['forecast']

        # 4.4.2 Loop until the thread is stopped
        start_time = time.perf_counter()
        samples = 0
        while self.running:

            # 4.4.2.1 Ensure consistent polling intervals
            if time.perf_counter() - start_time < samples * self.interval:
                time.sleep(samples * self.interval - (time.perf_counter() - start_time))

            # 4.4.2.2 Extract power readings and timestamp
            data = requests.get(self.url, timeout=2).json()
            timestamp = round(time.perf_counter(), 4)

            # 4.4.2.3 Extract power values for each target path
            readings = {}
            for descriptor, path in self.target_paths.items():
                val = get_node(data, path)['Value']
                m = num_re.search(val)
                power_value = float(m.group(1)) if m else 0.0
                readings[descriptor] = power_value

            # 4.4.2.4 Append the timestamped readings to the data list
            self.data.append((timestamp, readings))
            samples += 1
            
        # 4.4.3 Calculate average power and duration for later processing
        self.duration = self.data[-1][0] - self.data[0][0]
        for descriptor in self.target_paths:
            self.avg_power[descriptor] = np.mean([entry[1][descriptor] for entry in self.data])  # in W

# 5. Complete project
def run_project():

    # 5.1 Fetch initial data and determine power metric paths
    data = requests.get(power_url, timeout=2).json()
    power_metric_paths = walk(data)

    # 5.2 Measure the static usage of power for 10 seconds
    static_poll = PowerPoller(power_url, power_metric_paths, carbon_intensity_api)
    static_poll.start()
    time.sleep(15)
    static_poll.stop()

    # 5.3 Run the main function in continuous polling mode
    poller = PowerPoller(power_url, power_metric_paths, carbon_intensity_api)
    poller.start()
    starting_time = time.perf_counter()
    regression()
    recomposition()
    finishing_time = time.perf_counter()
    poller.stop()
    run_time = finishing_time - starting_time
    print(f"Total execution time: {run_time:.4f} seconds")

    # 5.4 Plot power consumption over time for CPU Package
    values1 = [entry[1]['CPU Package'] for entry in static_poll.data] + [entry[1]['CPU Package'] for entry in poller.data]
    values2 = [entry[1]['CPU Cores'] for entry in static_poll.data] + [entry[1]['CPU Cores'] for entry in poller.data]
    timestamps = [entry[0] - static_poll.data[0][0] for entry in static_poll.data] + [entry[0] - static_poll.data[0][0] for entry in poller.data]
    plt.close('all')
    plt.figure(figsize=(9,5))
    plt.plot(timestamps, values1, linewidth=3)
    plt.title('Energy Consumption of CPU Package', fontsize=18)
    plt.xlabel('Time Elapsed (seconds)', fontsize=16)
    plt.ylabel('Power Consumption (Watts)', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("Energy Consumption.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 5.5 Calculate carbon emissions
    ci = static_poll.carbon_intensity
    static_average = static_poll.avg_power['CPU Package']
    running_average = poller.avg_power['CPU Package']
    average_diff = running_average - static_average
    energy = running_average*run_time
    emission = (energy*ci/(3600*1000)) * 10**6

    print(f"Static Power: {static_average:.2f}, Running Power: {running_average:.2f}")
    print(f"At London's current carbon intensity of {ci} gCO2/kWh, with an average power of {running_average:.2f} W for {run_time:.2f}s, my code consumes {energy:.2f} J of energy and emits {emission:.2f} μg of CO2.\n")


# 6. Main execution
if __name__ == "__main__":
    run_project()