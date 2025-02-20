This repository is the capstone project for Data Science 4010 at Iowa State University. In this repository we analyze and organize data from a Dolt repo with the intention of modeling and visualizing volatility.

## Team Members
- Ryan Freidhoff (RyanFreidhoff)
- Nic Rhoads (nrhoads02)
- Dakota Rossi (PCodeXbro)
- Emiliano Saucedo (Emilianosau28)

## Folders
### Data
The data folder holds the raw data files for this project.
This project relies on two dolt repos to supply the base data.

Follow these steps to set up your local environment:

1. **Install Dolt**  
   Ensure that [Dolt](https://docs.dolthub.com/introduction/installation) is installed on your system. You can check your installation by running:
   ```bash
   dolt --version
   ```
   If Dolt isn’t installed, please follow the [installation instructions](https://docs.dolthub.com/introduction/installation) for your operating system.

2. **Clone the Repositories**  
   - **Clone the `stocks` repository:**  
     Run the following command to clone the **stocks** Dolt repository into `data/raw/stocks`:
     ```bash
     dolt clone https://www.dolthub.com/repositories/pcodexbro/stocks data/raw/stocks
     ```
   - **Clone the `options` repository:**  
     Similarly, clone the **options** Dolt repository into `data/raw/options`:
     ```bash
     dolt clone https://www.dolthub.com/repositories/pcodexbro/options data/raw/options
     ```

3. **Verify the Clones**  
   Confirm that the repositories have been cloned correctly by listing the contents of each directory:
   ```bash
   ls data/raw/stocks
   ls data/raw/options
   ```

The structure of this data is described further in [data/raw/README.md](data/raw/README.md) and [data/raw/METADATA.md](data/raw/METADATA.md)

From here, we will be building csv files through [src/data_collection/dolt_csv_export.py](src/data_collection/dolt_csv_export.py)
