import math
import csv
from decimal import Decimal

SELECTION_RATES = [0.10, 0.25, 0.5]
PRECISIONS = [0.65, 0.75, 0.85]
QUALIFICATION_RATES = [0.25, 0.50, 0.75]

def calculate_allocations(writer, N):
    for selection_rate in SELECTION_RATES:
        K = (int)(selection_rate*N)
        for precision in PRECISIONS:
            K_prime = (int)(precision*K)
            for qualification_rate in QUALIFICATION_RATES:
                N_prime = (int)(qualification_rate*N)
                equal_util_alloc = num_equal_allocations(N, K, N_prime, K_prime)
                total_alloc = math.comb(N, K)
                alloc_ratio = equal_util_alloc/total_alloc
                row = [N, K, N_prime, K_prime, selection_rate, precision, qualification_rate, equal_util_alloc, total_alloc, alloc_ratio]
                writer.writerow(row)

def num_equal_allocations(N, K, N_prime, K_prime):
    allocations = math.comb(N_prime, K_prime) * \
        math.comb(N - N_prime, K - K_prime)
    return Decimal(allocations)


if __name__ == "__main__":
    with open("synthetic_allocations.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["N", "K", "N'", "K'", "selection_rate", "precision", "qualification_rate", "equal_util_alloc", "total_alloc", "alloc_ratio"])
        calculate_allocations(writer, 100)
        calculate_allocations(writer, 1000)
