import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# Define input and output directories
input_folder = "results"
output_folder = "result_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot allocation experiment results for each unique (m, d, prob_d, partial_info_queries) combination.")
parser.add_argument("--x-axis", type=str, default="m", help="Column to use as the x-axis for each group")
args = parser.parse_args()

# Initialize a list to store the subplots (this will be combined later)
all_axes = []

# Iterate over each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Construct the full path to the file
        filepath = os.path.join(input_folder, filename)
        
        # Load data into a DataFrame
        data = pd.read_csv(filepath)
        
        # Ensure the specified x_axis column exists
        if args.x_axis not in data.columns:
            print(f"Column '{args.x_axis}' not found in {filename}. Skipping this file.")
            continue
        
        # Calculate the square root of variance columns to get standard deviations
        data['std_dev_n=m'] = np.sqrt(data['V(n=m)'])
        data['std_dev_n=m^2'] = np.sqrt(data['V(n=m^2)'])
        
        # Group by the combination of (m, d, prob_d, partial_info_queries)
        grouped_columns = [v for v in ['m', 'd', 'prob_d', 'batch', 'partial_info_queries'] if v != args.x_axis]
        grouped = data.groupby(grouped_columns)
        
        if all(len(group) <= 1 for dims, group in grouped):
            continue

        # Count total number of groups to create a suitable number of subplots
        total_plots = len(grouped)

        # Calculate number of rows and columns for the subplots grid
        rows = total_plots // 2 + (total_plots % 2 != 0)  # We want 2 plots per row
        cols = 2  # Two columns per row
        
        # Prepare the figure and axes only once for this file
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes = axes.flatten()  # Flatten the axes array to make it easier to index
        
        ymax = max(max(data['G(n=m)']), max(data['G(n=m^2)']), max(data['std_dev_n=m']), max(data['std_dev_n=m^2']) ) 
        ymin = min(min(data['G(n=m)']), min(data['G(n=m^2)']), min(data['std_dev_n=m']), min(data['std_dev_n=m^2']) ) 

        # Create a plot for each group
        for idx, (dims, group_data) in enumerate(grouped):
            ax = axes[idx]  # Use the current axis for this plot

            ax.set_title(", ".join([f"{col}: {dim}" for col, dim in zip(grouped_columns, dims)]))

            # Plot G(n=m) - Gap Mean for Light Load
            ax.plot(group_data[args.x_axis], group_data['G(n=m)'], marker='o', color='b', label='G(n=m)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            # Plot G(n=m^2) - Gap Mean for Heavy Load
            ax.plot(group_data[args.x_axis], group_data['G(n=m^2)'], marker='o', color='r', label='G(n=m^2)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            # Plot std_dev_n=m - Standard Deviation for Light Load
            ax.plot(group_data[args.x_axis], group_data['std_dev_n=m'], marker='s', color='g', label='std_dev(n=m)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            # Plot std_dev_n=m^2 - Standard Deviation for Heavy Load
            ax.plot(group_data[args.x_axis], group_data['std_dev_n=m^2'], marker='s', color='purple', label='std_dev(n=m^2)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            ax.set_ylim([ymin-ymax*0.05, ymax + ymax*0.05])

        # Adjust layout for all subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Define a single filename for all plots combined
        output_filename = f"{filename.split('.')[0]}_combined_by_{args.x_axis}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Save the figure with all subplots to the output folder
        plt.savefig(output_path)
        plt.close(fig)  # Close the figure to free memory

        print(f"Generated combined plot and saved as {output_path}")
