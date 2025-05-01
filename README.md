This repository is the capstone project for Data Science 4010 at Iowa State University. In this repository we analyze and organize data from a Dolt repo with the intention of modeling and visualizing volatility.

## Team Members
- Ryan Freidhoff (RyanFreidhoff)
- Nic Rhoads (nrhoads02)
- Dakota Rossi (dakota-rossi)
- Emiliano Saucedo (Emilianosau28)

## Application

Our dashboard can be found at [https://ds401-dern-volatility-dashboard.streamlit.app/](https://ds401-dern-volatility-dashboard.streamlit.app/)

## Folders

Below are descriptions of all essential folders for this project. There are other folders in the repo, but if they are not listed here they are not required for the application.

### Data

The data folder holds the raw data files for this project.
This project relies on two dolt repos to supply the base data.
You do not need to load the dolt repo, as we have generated parquet files that contain the same data, which can be found in the `parquet` subfolder.
We also have a `models` subfolder which stores our trained and compressed LGBM models used for our predictions and visualization.
For more information on our parquet split, we have [data/parquet/README.md](data/parquet/README.md).
For more information on our LGBM model compression and formatting, we have [data/models/surface_lgbm/README.md](data/models/surface_lgbm/README.md).
If you are interested in loading in the entire dolt repo, instructions can be found in [data/raw/README.md](data/raw/README.md).
Data structure is further described in [data/raw/METADATA.md](data/raw/METADATA.md).

### Dashboard

Contains our streamlit app files.

### Notebooks

Contains a few different notebooks that were used throughout development. These notebooks are pretty messy and should not be seen as a usage guide for our application, as they contain some old module versions, were used for training the models, etc, but may be helpful to understand our development process.

### SRC

Our source folder contains most of our main driver modules necessary for data extraction, transformation, loading, and model training. Specific READMEs can be found in each subfolder.
