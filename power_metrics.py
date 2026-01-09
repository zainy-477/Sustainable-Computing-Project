# power_metrics.py - extracts the power-related metrics from JSON data and returns their paths
# Zain Hirji, 14th December 2025

# 1. Imports
import requests
import re

# 2. Set desired locations and variables 
URL = "http://localhost:8085/data.json"
num_re = re.compile(r"(-?\d+(?:\.\d+)?)")

# 3. Walk function to traverse the JSON tree and extract all power-related metric paths
def walk(node, path="", power_metrics=None):

    # 3.1 Set up list to be filled with power metric paths
    if power_metrics is None:
        power_metrics = {}

    # 3.2 Obtain name, path, and value of the current node
    text = node.get("Text", "")
    path2 = f"{path}/{text}" if text else path
    val = node.get("Value")

    # 3.3 If the node's value ends with "W", extract and print the numeric part with its path
    if isinstance(val, str) and val.strip().endswith("W"):
        m = num_re.search(val)
        if m:
            print(f"{path2} = {m.group(1)} W")
            if float(m.group(1)) > 0.1:
                power_metrics[text] = path2

    # 3.4 Recursively walk through all child nodes
    for child in node.get("Children", []) or []:
        walk(child, path2, power_metrics)

    # 3.5 Return the dictionary of power metric paths
    return power_metrics

# 4. Main execution
if __name__ == "__main__":
    data = requests.get(URL, timeout=2).json()
    power_metrics = walk(data)