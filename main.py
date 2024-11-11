import numpy as np
import argparse
import pandas as pd
from multiprocessing import Process, Queue, Pool
import itertools
from copy import copy

def max_gap(bins: np.array) -> np.float32:
    return np.max(bins-np.mean(bins))

def run_experiment(
    seed: int, 
    m: int, 
    d: int, 
    prob_d: float,
    batch: int,
    partial_info_queries: int
):
    np.random.seed()
    n = max(m**2//batch, 1)
    bins = np.zeros(m)
    bins_indices = np.array(range(len(bins)))
    max_gap_n_eq_m = 0
    for b in range(0, n):
        batch_bins = copy(bins)
        for i in range(0, min(batch, m**2)):
            option_bins = np.random.choice(
                bins_indices, 
                size=d if np.random.random_sample() < prob_d else 1, 
                replace=True
            )
            if partial_info_queries == 0:
                aux_bins = bins[option_bins]
                option_bins = option_bins[aux_bins == aux_bins.min()]
            else:
                percentile = 50
                for j in range(partial_info_queries):
                    if len(option_bins) < 2:
                        break
                    bins_perc_value = np.percentile(bins, percentile)
                    lt_median_option_bins = option_bins[bins[option_bins] < bins_perc_value]
                    if len(lt_median_option_bins) > 0:
                        option_bins = lt_median_option_bins
                        percentile -= 100/(2**(j+2))
                    else:
                        percentile += 100/(2**(j+2))
            chosen_bin = np.random.choice(option_bins)
            batch_bins[chosen_bin] += 1
            if b*batch+i == m-1:
                max_gap_n_eq_m = max_gap(batch_bins)
        bins = batch_bins
    return max_gap_n_eq_m, max_gap(bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=str, help="Number of bins")
    parser.add_argument("--option-bins", type=str, help="Number of option bins for a ball to be added")
    parser.add_argument("--reps", type=int, help="Number of repetition", default=10)
    parser.add_argument("--probs-d", type=str, help="Probability of using d option bins instead of 1 option bin", default="[1.0]")
    parser.add_argument("--batches", type=str, help="The number of balls that will be assigned with the same information in the bins", default="[1]")
    parser.add_argument("--partial-info-queries", type=str, help="How many partial information queries to execute, 0 for full info", default="[0]")
    parser.add_argument("--result-name", type=str, default="result.csv")
    args = parser.parse_args()

    t = int(max(args.reps, 1))
    bins = eval(args.bins)
    option_bins = eval(args.option_bins)
    probs = eval(args.probs_d)
    batches = eval(args.batches)
    partial_info_queries = eval(args.partial_info_queries)

    experiments = []
    for (m, d, prob_d, batch, piq) in itertools.product(bins, option_bins, probs, batches, partial_info_queries):
        m = int(max(m, 1))
        d = int(max(d, 1))
        prob_d = min(max(prob_d, 0.0), 1.0)
        batch = int(max(batch, 1))
        piq = int(max(piq, 0))

        with Pool() as pool: 
            results = [pool.apply_async(run_experiment, (i, m, d, prob_d, batch, piq))
                    for i in range(t)]
            results = [result.get() for result in results]
            linear_n_results = [result[0] for result in results]
            big_n_results = [result[1] for result in results]

        linear_n_gap_mean = np.mean(np.asarray(linear_n_results))
        linear_n_variance = np.var(np.asarray(linear_n_results))
        big_n_gap_mean = np.mean(np.asarray(big_n_results))
        big_n_gap_variance = np.var(np.asarray(big_n_results))

        experiments.append({
            "m": m,
            "d": d,
            "T": t,
            "prob_d": prob_d,
            "batch": batch,
            "partial_info_queries": piq,
            "G(n=m)": linear_n_gap_mean,
            "G(n=m^2)": big_n_gap_mean,
            "V(n=m)": linear_n_variance,
            "V(n=m^2)": big_n_gap_variance
        })

        print(list(experiments[-1].keys()))
        print(list(experiments[-1].values()))

    pd.DataFrame(experiments).to_csv("results/"+args.result_name+".csv")

if __name__ == "__main__":
    main()
