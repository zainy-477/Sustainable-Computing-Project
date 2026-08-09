**Description**

The aim of this project is to forecast UK carbon intensity (CI) in 2025 in a sustainable manner, given training data from 2009-2024. The dataset used df\_fuel\_ckan.csv was obtained from NESO. There are three methods used; recomposition (non-ML), nonlinear regression (ML), and XGBoost (ML). For each method, there is a function to forecast CI at a monthly resolution and at an hourly resolution.



To run each method on its own, simply execute the corresponding file (recomposition_method.py, regression_method.py, XGBoost_method.py). Only the recomposition and nonlinear regression methods are used in final analysis, which takes place in power\_meter.py



**Sustainable Analysis**

Each function records its own runtime, and reports this alongside training and validation results. The main execution file is power\_meter.py, which measures the power consumption of overall code execution. It also calculates the carbon emissions of the code, using real-time CI values in London that are obtained from Carbon Intensity API.



**Visualisations and Results**

Each function produces some graphs for insight, however the majority of analysis is done in the Jupyter Notebook archived\_analysis.ipynb. These include context for the dataset, and prior regression models that were discarded. The overall set of results for the project can be found in Results.xlsx. Relevent images are found in the Images folder.

The written report for the project can be found at Project Report.pdf

