Execution Guide: From Zero to Simulation (Phase 1)
This document describes the complete process for setting up the simulation environment and running the Phase 1 experiments from a clean machine.

Step 1: Environment Setup

These steps only need to be performed once per machine.

1.1. Clone the Repository

First, download the source code from the GitHub repository.

git clone https://github.com/cesaragostino/DOFT-Delayed-Oscillator-Field-Theory.git
cd DOFT-Delayed-Oscillator-Field-Theory

1.2. Create the Conda Environment

The project uses a Conda environment to manage all Python dependencies. The environment.yml file in the project's root directory defines everything needed.

# This command will read the file and create an environment named 'doft_v12'
conda env create -f environment.yml

1.3. Activate the Environment

Once created, activate the environment to use the installed tools.

conda activate doft_v12

Note: You will need to run this command every time you open a new terminal to work on the project.

Step 2: Update Source Code for Phase 1

Before running, you must ensure your local files match the code provided for the Phase 1 experiments.

2.1. Update Simulator Code

Replace the content of the following files with the code I provided previously:

src/doft/model.py

src/doft/run_sim.py

2.2. Create Configuration and Runner Script

The following files need to be created.

Create the configuration file configs/config_phase1.json.

Create the execution script scripts/run_phase1.sh.

Make sure their content is copied exactly from the code blocks I provided.

Step 3: Running the Phase 1 Experiments

With the environment set up and the files updated, you can now launch the simulations.

3.1. The Main Script

The scripts/run_phase1.sh script is the single entry point you need. It handles setting variables, checking the environment, and launching the Python simulator with the correct Phase 1 configuration.

3.2. CPU Execution

To run all Phase 1 simulations using your machine's CPU cores:

# N_JOBS=4 will use 4 processes in parallel. Adjust this to the number of cores on your machine.
N_JOBS=4 bash scripts/run_phase1.sh

3.3. GPU Execution

If your machine has a CUDA-compatible NVIDIA GPU and you have installed the PyTorch wheels, you can accelerate the simulation as follows:

# N_JOBS is typically kept at 1.
N_JOBS=1 bash scripts/run_phase1.sh

The script will verify if a GPU is available. If not, it will notify you and proceed with the execution on the CPU.

Step 4: Results

Upon completion, the script will create a new directory inside the /results folder. The directory name will include the date and time of the run, for example: results/phase1_run_20250825_183000.

Inside that folder, you will find the simulation artifacts (runs.csv, blocks.csv, etc.), ready for analysis.

