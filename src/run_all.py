"""Run the full pipeline: dataset generation and tasks 1-5.

Run:  python src/run_all.py
"""
import time

import generate_dataset
import task1_check_setup
import task2_eda
import task3_ml_models
import task4_deep_learning
import task5_ethics

STEPS = [
    ("Dataset generation", generate_dataset.main),
    ("Task 1", task1_check_setup.main),
    ("Task 2", task2_eda.main),
    ("Task 3", task3_ml_models.main),
    ("Task 4", task4_deep_learning.main),
    ("Task 5", task5_ethics.main),
]


def main() -> None:
    start = time.perf_counter()
    for name, step in STEPS:
        print("\n" + "=" * 70)
        step()
    print("\n" + "=" * 70)
    print(f"All steps finished in {time.perf_counter() - start:.1f} s. "
          "See figures/ and results/.")


if __name__ == "__main__":
    main()
