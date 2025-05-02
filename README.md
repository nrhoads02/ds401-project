# DS 4010 Volatility Surface Capstone

This repository contains the capstone project for Data Science 4010 at Iowa State University (Spring 2024). We analyze and organize financial data, primarily sourced from Dolt databases, to model and visualize stock volatility surfaces.

## Team Members

* Ryan Freidhoff (RyanFreidhoff)
* Nic Rhoads (nrhoads02)
* Dakota Rossi (dakota-rossi)
* Emiliano Saucedo (Emilianosau28)

## Application Dashboard

Our interactive Streamlit dashboard demonstrating the volatility surface modeling can be accessed here:
**[Volatility Dashboard](https://ds401-dern-volatility-dashboard.streamlit.app/)**

## Repository Structure

Below are descriptions of the essential folders within this project. Other folders may exist but are not critical for the core application or data processing pipeline.

---

### [`data/`](data/)

This directory houses all data used in the project.

* **Source Data:** The project initially utilizes two Dolt repositories (`data/raw/stocks` and `data/raw/options`) as the primary data source.
    * **Note:** Cloning and using the Dolt repositories directly is **not required** for most use cases.
    * Instructions for Dolt setup (if desired): [`data/raw/README.md`](data/raw/README.md)
    * Raw data schema details: [`data/raw/METADATA.md`](data/raw/METADATA.md)
* **Processed Data (`data/parquet/`):** To facilitate efficient access and manage repository size, the core datasets (OHLCV, options chain, splits) have been converted to partitioned Parquet files. This is the **recommended** way to access the data.
    * Details on Parquet structure and partitioning: [`data/parquet/README.md`](data/parquet/README.md)
* **Models (`data/models/`):** Contains the pre-trained and compressed LightGBM models used for realized volatility predictions.
    * Details on model structure and loading: [`data/models/surface_lgbm/README.md`](data/models/surface_lgbm/README.md)

---

### [`dashboard/`](dashboard/)

Contains all necessary files for the Streamlit web application, including the main application script and supporting pages.

---

### [`notebooks/`](notebooks/)

Includes Jupyter notebooks used during development for exploration, experimentation, model training, and testing.

* **Note:** These notebooks may contain outdated code versions or complex procedures (like model training) and are primarily for understanding the development process, not as a direct usage guide for the final application or modules.

---

### [`src/`](src/)

This is the main source code directory containing Python modules for the data pipeline and modeling.

* **Data Extraction (`src/data_extraction/`):** Modules for extracting data from sources (Dolt, CSV) and loading the processed Parquet files.
    * Details: [`src/data_extraction/README.md`](src/data_extraction/README.md)
* **Data Transformation (`src/data_transformation/`):** Modules for cleaning data, performing split adjustments, calculating technical indicators, and joining auxiliary data.
    * Details: [`src/data_transformation/README.md`](src/data_transformation/README.md)
* **Data Modeling (`src/data_modeling/`):** Modules for training volatility models (LightGBM), generating predictions, analyzing option chain data, and visualizing volatility surfaces.
    * Details: [`src/data_modeling/README.md`](src/data_modeling/README.md)

---
