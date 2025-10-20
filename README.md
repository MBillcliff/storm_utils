# storm_utils

## Introduction

This repository provides shared utilities and notebooks for geomagnetic storm forecasting and regression using solar wind ensembles. It includes:

- Functions for downloading and processing OMNI and Hpo data
- Routines for running ambient HUXt ensembles using [HUXt](https://github.com/University-of-Reading-Space-Science/HUXt) and [HUXt_tools](https://github.com/mathewjowens/HUXt_tools)
- Preprocessing notebooks for generating data used in both the classification and regression projects

## Installation

Navigate to a common directory e.g. ~/storm_project where you'd like to store all storm-related repositories, then clone the required ones:
```bash
git clone https://github.com/MBillcliff/storm_utils
git clone https://github.com/University-of-Reading-Space-Science/HUXt
git clone https://github.com/mathewjowens/HUXt_tools
```
Clone either or both analysis repositories:
```bash
git clone https://github.com/MBillcliff/storm_classification
git clone https://github.com/MBillcliff/storm_regression
```
Note: storm_utils is intended to be installed as an editable package from each project environment (see usage instructions in the classification or regression repositories).

All notebooks in storm_utils/notebooks/ are designed to be run from the environment of either [storm_classification](https://github.com/MBillcliff/storm_classification) or [storm_regression](https://github.com/MBillcliff/storm_regression). You do not need to create a separate environment for storm_utils.

## Usage

The following notebooks are available in ```notebooks/```:

[```data_downloading.ipynb```](https://github.com/MBillcliff/storm_utils/notebooks/data_downloading.ipynb)
Downloads solar wind and geomagnetic index data.

[```ambient_huxt.ipynb```](https://github.com/MBillcliff/storm_utils/notebooks/ambient_huxt.ipynb)
Runs ambient HUXt ensemble simulations and prepares HUXt ensemble output data for use in downstream ML pipelines.


## Contact
Please contact [Matthew Billcliff](https://github.com/MBillcliff/).