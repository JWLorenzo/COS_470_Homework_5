# Name: Jacob Lorenzo
# Assignment: Homework 5
# Class: COS 470/570
# Instructor: Dr. Hutchinson
# Date: 11/22/24 -


import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import math
import pandas as pd
import numpy as np
import os
import sys


def load_Dataset(filename: str) -> pd.DataFrame:
    dir_Path = os.getcwd()
    if os.path.exists(os.path.join(dir_Path, filename)) or os.path.exists(filename):
        dataset = pd.read_csv(filename)
        return dataset
    else:
        print(f"Invalid Path! {os.path.join(dir_Path, filename)} nor {filename} exist.")
        sys.exit(1)


def get_Trainingset(
    dataset: pd.DataFrame,
    compareSet: pd.DataFrame,
    trainingPercent: float,
    trainingYears: list[int, int],
) -> pd.DataFrame:
    if not trainingPercent:
        trainingset = dataset[
            (dataset["YEAR"] != trainingYears[0])
            & (dataset["YEAR"] != trainingYears[1])
        ]
        return trainingset.drop(columns=["YEAR", "TEAM"])
    else:
        return dataset[~dataset.index.isin(compareSet.index)].drop(
            columns=["YEAR", "TEAM"]
        )
        # Implement a difference based on the test set


def get_Testset(
    dataset: pd.DataFrame, trainingPercent: float, trainingYears: list[int, int]
) -> pd.DataFrame:
    if not trainingPercent:
        testset = dataset[
            (dataset["YEAR"] == trainingYears[0])
            | (dataset["YEAR"] == trainingYears[1])
        ]
        return testset.drop(columns=["YEAR", "TEAM"])
    else:
        return dataset.sample(frac=trainingPercent).drop(columns=["YEAR", "TEAM"])
        # create a dataframe that is a random selection


def get_Headers(dataset: pd.DataFrame) -> list:
    header = list(dataset)
    return header


def create_Datasets(
    training_set: pd.DataFrame, test_set: pd.DataFrame, batch_size
) -> list[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_Dataset = CustomDataset(training_set)
    test_Dataset = CustomDataset(test_set)

    train_Loader = torch.utils.data.DataLoader(
        train_Dataset, batch_size=batch_size, shuffle=True
    )
    test_Loader = torch.utils.data.DataLoader(
        test_Dataset, batch_size=batch_size, shuffle=False
    )

    return train_Loader, test_Loader


# W,G,EFG,EFGD,TOR,TORD,ORB,DRB,FTR,FTRD,2P_O,2P_D,3P_O,3P_D,ADJ_T,YEAR,TEAM


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = torch.tensor(dataframe.iloc[:, 1:].values)
        self.targets = torch.tensor(dataframe.iloc[:, 0].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Homework_5", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        type=str,
        default="homework5_data.csv",
        help="Dataset Path",
    )

    parser.add_argument(
        "-p",
        "--percent",
        nargs="?",
        type=float,
        const=0.10,
        default=None,
        help="How much of the dataset split to training",
    )
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        default=[2008, 2024],
        help="How much of the dataset split to training",
    )

    args = parser.parse_args()
    dataset = load_Dataset(args.file)
    TRAINING_SPLIT_PERCENT = args.percent
    TRAINING_SPLIT_YEARS = random.choices(range(args.years[0], args.years[1] + 1), k=2)
    CSV_HEADER = get_Headers(dataset)
    TEST_SET = get_Testset(dataset, TRAINING_SPLIT_PERCENT, TRAINING_SPLIT_YEARS)
    TRAINING_SET = get_Trainingset(
        dataset, TEST_SET, TRAINING_SPLIT_PERCENT, TRAINING_SPLIT_YEARS
    )

    print(CSV_HEADER)
    print(TRAINING_SET.sample(10))
    print(TEST_SET.sample(10))


if __name__ == "__main__":
    main()
