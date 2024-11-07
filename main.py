import numpy as np
import argparse
from multiprocessing import Process, Queue, Pool

def max_gap(bins: np.array) -> np.float32:
    return np.max(bins-np.mean(bins))

def run_experiment(
    seed: int, 
    m: int, 
    d: int, 
    prob_d: float
):
    np.random.seed()
    n = m**2
    bins = np.zeros(m)
    bins_indices = np.array(range(len(bins)))
    max_gap_n_eq_m = 0
    for i in range(n):
       option_bins = np.random.choice(
            bins_indices, 
            size=d if np.random.random_sample() < prob_d else 1, 
            replace=True
        )
       chosen_bin = option_bins[np.argmin(bins[option_bins])]
       bins[chosen_bin] += 1
       if i == m:
           max_gap_n_eq_m = max_gap(bins)
           #print(bins)
           #print(max_gap(bins))

    #print(bins)
    #print(max_gap(bins))
    return max_gap_n_eq_m, max_gap(bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, help="Number of bins")
    parser.add_argument("d", type=int, help="Number of option bins for a ball to be added")
    parser.add_argument("T", type=int, help="Number of repetition")
    parser.add_argument("--prob-d", type=float, help="Probability of using d option bins instead of 1 option bin")
    args = parser.parse_args()
    m = args.m
    d = args.d
    t = args.T
    prob_d = args.prob_d if not args.prob_d is None else 1.0
    if not isinstance(d, int):
        raise ValueError("d must result in an 'int'")
    
    """
    lin_result, fin_result = 0, 0
    for i in range(t):
        lin_mg, fin_mg = run_experiment(i, m, d)
        lin_result += lin_mg/t
        fin_result += fin_mg/t

    print(f"(n=m) gap mean of {t}: {lin_result}")
    print(f"(n=m^2) gap mean of {t}: {fin_result}")
    """

    with Pool() as pool: 
        results = [pool.apply_async(run_experiment, (i, m, d, prob_d))
                   for i in range(t)]
        results = [result.get() for result in results]
        linear_n_results = [result[0] for result in results]
        big_n_results = [result[1] for result in results]

        linear_n_gap_mean = np.mean(np.asarray(linear_n_results))
        big_n_gap_mean = np.mean(np.asarray(big_n_results))

    print(f"(n=m) gap mean of {t}: {linear_n_gap_mean}")
    print(f"(n=m^2) gap mean of {t}: {big_n_gap_mean}")


if __name__ == "__main__":
    main()
