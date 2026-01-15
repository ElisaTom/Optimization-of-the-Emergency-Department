# Optimization-of-the-Emergency-Department
This project implements an MLOps pipeline for Process Mining in a healthcare context. It ingests hospital event logs, cleans the data, and uses PM4Py (Process Mining for Python) to visualize patient journeys (Process Discovery).

The project is containerized using Docker to ensure reproducibility and uses DVC for pipeline management.
Before you begin, ensure you have the following installed on your machine:

Git

Docker Desktop (running)

- Installation & Setup:
1. Clone the Repository
Open your terminal and clone the project:
git clone <https://github.com/annapierroo/Optimization-of-the-Emergency-Department.git>

2. Add the Dataset
The raw data is not included in the repository for size reasons.

Create the directory if it doesn't exist: data/raw/

Place your procedures.csv file inside data/raw/.

3. Build the Docker Image
Build the container that holds all the necessary libraries (Pandas, PM4Py, etc.):

docker build -t mlops-hospital .


- How to Run the Analysis
You can run the entire pipeline (Ingestion + Discovery) using a single Docker command. This will mount your current folder so the output images are saved directly to your local machine.

Option A: Run the Full Pipeline (One-Liner)
This command runs the data ingestion, generates the graph, and fixes file permissions so you can view the result.

docker run -v $(pwd):/app mlops-hospital /bin/bash -c "python src/ingest_data.py && python src/process_discovery.py && chmod -R 777 reports"

Option B: Run via DVC (Data Version Control)
If you are using DVC to manage the steps defined in dvc.yaml:

docker run -v $(pwd):/app mlops-hospital /bin/bash -c "dvc repro -f && chmod -R 777 reports"

- Customization
Changing the Number of Patients
To analyze a larger or smaller cohort, edit the file src/ingest_data.py:

# Inside src/ingest_data.py
n_cases = 500  # <--- Change this number
target_ids = df['case:concept:name'].unique()[:n_cases]

- Outputs
After running the pipeline, check the reports/figures/ folder. You will find the generated process maps.