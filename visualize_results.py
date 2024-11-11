import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

input_folder = "results"
output_folder = "result_images"

os.makedirs(output_folder, exist_ok=True)

parser = argparse.ArgumentParser(description="Plot allocation experiment results for each unique (m, d, prob_d, partial_info_queries) combination.")
parser.add_argument("--x-axis", type=str, default="m", help="Column to use as the x-axis for each group")
args = parser.parse_args()

all_axes = []

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        
        data = pd.read_csv(filepath)
        
        if args.x_axis not in data.columns:
            print(f"Column '{args.x_axis}' not found in {filename}. Skipping this file.")
            continue
        
        data['std_dev_n=m'] = np.sqrt(data['V(n=m)'])
        data['std_dev_n=m^2'] = np.sqrt(data['V(n=m^2)'])
        
        grouped_columns = [v for v in ['m', 'd', 'prob_d', 'batch', 'partial_info_queries'] if v != args.x_axis]
        grouped = data.groupby(grouped_columns)
        
        if all(len(group) <= 1 for dims, group in grouped):
            continue

        total_plots = len(grouped)

        rows = total_plots // 2 + (total_plots % 2 != 0)  
        cols = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes = axes.flatten()  
               
        ymax = max(max(data['G(n=m)']), max(data['G(n=m^2)']), max(data['std_dev_n=m']), max(data['std_dev_n=m^2']) ) 
        ymin = min(min(data['G(n=m)']), min(data['G(n=m^2)']), min(data['std_dev_n=m']), min(data['std_dev_n=m^2']) ) 

        for idx, (dims, group_data) in enumerate(grouped):
            ax = axes[idx]

            ax.set_title(", ".join([f"{col}: {dim}" for col, dim in zip(grouped_columns, dims)]))

            ax.plot(group_data[args.x_axis], group_data['G(n=m)'], marker='o', color='b', label='G(n=m)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            ax.plot(group_data[args.x_axis], group_data['G(n=m^2)'], marker='o', color='r', label='G(n=m^2)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            ax.plot(group_data[args.x_axis], group_data['std_dev_n=m'], marker='s', color='g', label='std_dev(n=m)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            ax.plot(group_data[args.x_axis], group_data['std_dev_n=m^2'], marker='s', color='purple', label='std_dev(n=m^2)')
            ax.set_xlabel(args.x_axis)
            ax.legend()

            ax.set_ylim([ymin-ymax*0.05, ymax + ymax*0.05])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_filename = f"{filename.split('.')[0]}_combined_by_{args.x_axis}.png"
        output_path = os.path.join(output_folder, output_filename)

        plt.savefig(output_path)
        plt.close(fig) 

        print(f"Generated combined plot and saved as {output_path}")
