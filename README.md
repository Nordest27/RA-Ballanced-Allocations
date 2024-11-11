README.md
# Balanced Allocations

## Requirements

This project requires:
- Python 3.8+
- NumPy
- Pandas
- Matplotlib

## Installation

1. **Clone the repository**:
   git clone https://github.com/Nordest27/RA-Ballanced-Allocations.git
   cd RA-Ballanced-Allocations
Install dependencies: You can install the required packages using pip and the provided requirements.txt file:

pip install -r requirements.txt

Usage

Running Experiments

The main script, main.py, runs the experiments based on different allocation strategies and parameters. Here is an example usage:

python3 main.py --bins "[10 * i for i in range(1, 11)]" --option-bins "[2]" --reps 100 --batches "[100 * i for i in range(1, 72, 7)]" --result-name "2_d_choice_with_batches"

--bins: Specifies the range of bins used in the experiments. This example uses [10, 20, 30, ..., 100].

--option-bins: Number of bins chosen for each ball placement (e.g., 2 for 2-choice).

--reps: Number of repetitions for each experiment (e.g., 100).

--batches: Specifies batch sizes, simulating outdated information; here [100, 800, 1500, ..., 7000].

--result-name: Specifies the output filename for results in .csv format, saved in the results/ directory.
Results are saved as CSV files with mean and variance for each experiment, labeled by the parameters used.

Visualizing Results

To generate plots of the experimental results, use visualize_results.py with the appropriate axis parameter:

python3 visualize_results.py --x-axis m

--x-axis: Parameter for the x-axis of the generated plots (e.g., m, representing the number of bins).
This script loads results from the results/ directory and generates visualizations, including mean gaps and variances for different scenarios.

