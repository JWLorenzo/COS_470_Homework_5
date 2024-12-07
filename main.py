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
import pandas as pd
import numpy as np
import os
import sys
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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
    scaler: StandardScaler,
) -> pd.DataFrame:
    if not trainingPercent:
        trainingset = dataset[
            (dataset["YEAR"] != trainingYears[0])
            & (dataset["YEAR"] != trainingYears[1])
        ]
    else:
        trainingset = dataset[~dataset.index.isin(compareSet.index)]
    # Normalize
    features = trainingset.iloc[:, 2:13]
    trainingset.iloc[:, 2:13] = scaler.transform(features)

    return trainingset.drop(columns=["YEAR", "TEAM"])


def get_Testset(
    dataset: pd.DataFrame, trainingPercent: float, trainingYears: list[int, int]
) -> tuple[pd.DataFrame, StandardScaler]:
    if not trainingPercent:
        testset = dataset[
            (dataset["YEAR"] == trainingYears[0])
            | (dataset["YEAR"] == trainingYears[1])
        ]
    else:
        testset = dataset.sample(frac=trainingPercent)
        # create a dataframe that is a random selection

    scaler = StandardScaler()
    features = testset.iloc[:, 2:13]
    testset.iloc[:, 2:13] = scaler.fit_transform(features)
    return testset.drop(columns=["YEAR", "TEAM"]), scaler


def create_Datasets(dataset: pd.DataFrame, batch_size) -> data_utils.DataLoader:
    loaded_Dataset = CustomDataset(dataset)

    dataset_Loader = data_utils.DataLoader(
        dataset=loaded_Dataset, batch_size=batch_size, shuffle=True
    )

    return dataset_Loader


# W,G,EFG,EFGD,TOR,TORD,ORB,DRB,FTR,FTRD,2P_O,2P_D,3P_O,3P_D,ADJ_T,YEAR,TEAM


class CustomDataset(data_utils.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = torch.tensor(dataframe.iloc[:, 1:13].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def parse_Arguments():
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
    parser.add_argument(
        "-b",
        "--batch",
        nargs="?",
        type=int,
        const=64,
        default=64,
        help="Batch Size",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        nargs="?",
        type=int,
        const=100,
        default=100,
        help="Number of epochs",
    )

    parser.add_argument(
        "-l",
        "--learn",
        nargs="?",
        type=float,
        const=0.0001,
        default=0.0001,
        help="Learning rate",
    )

    return parser.parse_args()


def train(
    dataloader: data_utils.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    # time for a training montage!
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: data_utils.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    num_batches = len(dataloader)
    test_loss = 0

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            all_preds.append(pred.cpu().detach().numpy())
            all_true.append(y.cpu().detach().numpy())

            test_loss += loss_fn(pred, y.view(-1, 1)).item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    test_loss /= num_batches

    r2 = r2_score(all_true, all_preds)

    print(f"Test MSE: {test_loss:>8f} R²: {r2:.4f}")


def main() -> None:
    ARGS = parse_Arguments()
    DATASET = load_Dataset(ARGS.file)
    BATCH_SIZE = ARGS.batch
    TRAINING_SPLIT_PERCENT = ARGS.percent
    TRAINING_SPLIT_YEARS = random.choices(range(ARGS.years[0], ARGS.years[1] + 1), k=2)
    TEST_FRAME, SCALER = get_Testset(
        DATASET, TRAINING_SPLIT_PERCENT, TRAINING_SPLIT_YEARS
    )
    TRAINING_FRAME = get_Trainingset(
        DATASET, TEST_FRAME, TRAINING_SPLIT_PERCENT, TRAINING_SPLIT_YEARS, SCALER
    )
    EPOCHS = ARGS.epochs
    LEARNING_RATE = ARGS.learn

    train_Loader = create_Datasets(TRAINING_FRAME, BATCH_SIZE)
    test_Loader = create_Datasets(TEST_FRAME, BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    MODEL = nn.Sequential(
        nn.Linear(12, BATCH_SIZE),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE, BATCH_SIZE),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE, 1),
    ).to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)

    # Simulating training

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n")
        train(train_Loader, MODEL, loss_function, optimizer, device)
        test(test_Loader, MODEL, loss_function, device)
    print("Finished!")


if __name__ == "__main__":
    main()
