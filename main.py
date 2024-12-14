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
import matplotlib.pyplot as plt


# Function to load the dataset from a specified file
def load_Dataset(filename: str) -> pd.DataFrame:
    dir_Path = os.getcwd()  # Get current working directory
    if os.path.exists(os.path.join(dir_Path, filename)) or os.path.exists(filename):
        dataset = pd.read_csv(filename)  # Read the CSV file into a pandas DataFrame
        return dataset
    else:
        print(f"Invalid Path! {os.path.join(dir_Path, filename)} nor {filename} exist.")
        sys.exit(1)  # Exit if the file doesn't exist


# Function to split the dataset into training and testing sets
def get_Datasets(
    dataset: pd.DataFrame,
    trainingPercent: float,
    trainingSchool: float,
    trainingYears: list[int, int],
    save: bool,
    trainFile: str,
    testFile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schoolset = dataset["TEAM"].unique()  # Get unique schools from the dataset
    schoolList = []  # List of schools to be used for testing
    if (testFile is not None and os.path.exists(testFile)) and (
        trainFile is not None and os.path.exists(trainFile)
    ):
        # If train and test files are provided, load them
        if not trainingPercent:
            # Split based on years if no percent is specified
            testset = dataset[
                (dataset["YEAR"] == trainingYears[0])
                | (dataset["YEAR"] == trainingYears[1])
            ]
            trainingset = dataset[
                (dataset["YEAR"] != trainingYears[0])
                & (dataset["YEAR"] != trainingYears[1])
            ]
        elif trainingSchool:
            # Split based on a percentage of schools
            schoolList = random.sample(
                schoolset, k=round(len(schoolset) * trainingSchool)
            )
            testset = dataset[dataset["TEAM"].isin(schoolList)]
            trainingset = dataset[~dataset["NAME"].isin(schoolList)]
        else:
            # Split by a flat percentage
            testset = dataset.sample(frac=trainingPercent)
            trainingset = dataset[~dataset.index.isin(testset.index)]

        # Optionally save the datasets to CSV files
        if save:
            testset.to_csv("testing.csv", index=False)
            trainingset.to_csv("training.csv", index=False)
    else:
        # If no files are provided, load the datasets
        testset = load_Dataset(testFile)
        trainingset = load_Dataset(trainFile)

    # Normalize the features of the training and test datasets
    scaler = StandardScaler()  # Create a StandardScaler object
    train_features = trainingset.iloc[:, 2:13]  # Select feature columns
    test_features = testset.iloc[:, 2:13]  # Select feature columns
    trainingset.iloc[:, 2:13] = scaler.fit_transform(
        train_features
    )  # Fit and transform training data
    testset.iloc[:, 2:13] = scaler.fit_transform(test_features)  # Transform test data

    # Drop non-feature columns ('YEAR' and 'TEAM') before returning
    return (
        trainingset.drop(columns=["YEAR", "TEAM"]),
        testset.drop(columns=["YEAR", "TEAM"]),
    )


# Function to create DataLoader from a pandas DataFrame
def create_Datasets(dataset: pd.DataFrame, batch_size) -> data_utils.DataLoader:
    loaded_Dataset = CustomDataset(dataset)  # Create a custom dataset

    # Create a DataLoader to handle batching and shuffling
    dataset_Loader = data_utils.DataLoader(
        dataset=loaded_Dataset, batch_size=batch_size, shuffle=True
    )

    return dataset_Loader


# Custom Dataset class to handle the data in batches
class CustomDataset(data_utils.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        # Convert the features and targets into tensors
        self.data = torch.tensor(dataframe.iloc[:, 1:13].values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.float32)

    def __len__(self):
        # Return the length of the dataset (number of samples)
        return len(self.data)

    def __getitem__(self, index):
        # Return the data and target at the given index
        return self.data[index], self.targets[index]


# Function to parse command line arguments
def parse_Arguments():
    parser = argparse.ArgumentParser(
        prog="Homework_5", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Define arguments for dataset paths, hyperparameters, and plot options
    parser.add_argument(
        "-f",
        "--file",
        nargs="?",
        type=str,
        default="homework5_data.csv",
        help="Dataset Path",
    )
    parser.add_argument(
        "-te", "--test", nargs="?", type=str, default="None", help="Test Dataset Path"
    )
    parser.add_argument(
        "-tr", "--train", nargs="?", type=str, default="None", help="Train Dataset Path"
    )
    parser.add_argument(
        "-sv",
        "--save",
        action="store_true",
        default=False,
        help="Save Datasets to Files",
    )
    parser.add_argument(
        "-p",
        "--percent",
        nargs="?",
        type=float,
        const=0.10,
        default=None,
        help="Training Split Percentage",
    )
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        default=[2008, 2024],
        help="Training Years",
    )
    parser.add_argument(
        "-b", "--batch", nargs="?", type=int, const=64, default=64, help="Batch Size"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        nargs="?",
        type=int,
        const=200,
        default=200,
        help="Number of Epochs",
    )
    parser.add_argument(
        "-l",
        "--learn",
        nargs="?",
        type=float,
        const=0.00001,
        default=0.00001,
        help="Learning Rate",
    )
    parser.add_argument(
        "-pl", "--plot", action="store_true", default=False, help="Generate Plots?"
    )
    parser.add_argument(
        "-s",
        "--school",
        nargs="?",
        type=float,
        const=0.10,
        default=None,
        help="Percentage of Schools for Testing",
    )

    return parser.parse_args()  # Return the parsed arguments


# Function to train the model for one epoch
def train(
    dataloader: data_utils.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> list[float]:
    size = len(dataloader.dataset)
    current = 0
    loss_list = []  # List to store loss values for each batch
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)  # Model prediction
        loss = loss_fn(pred, y.view(-1, 1))  # Calculate the loss

        optimizer.zero_grad()  # Reset the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model's parameters
        loss, current = loss.item(), batch * len(x)
        loss_list.append(loss)

        if batch % 10 == 0:
            print(
                f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]"
            )  # Print loss every 10 batches

    return loss_list  # Return the loss for each batch


# Function to test the model
def test(
    dataloader: data_utils.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    num_batches = len(dataloader)
    test_loss = 0

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            all_preds.append(pred.cpu().detach().numpy())  # Store predictions
            all_true.append(y.cpu().detach().numpy())  # Store true values

            test_loss += loss_fn(pred, y.view(-1, 1)).item()  # Accumulate loss

    # Calculate final average loss and R² score
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    test_loss /= num_batches

    r2 = r2_score(all_true, all_preds)  # Calculate R² score

    print(f"Test MSE: {test_loss:>8f} R²: {r2:.4f}")  # Print the results
    return test_loss, r2


# Function to create a line plot
def create_Plot(
    x_axis: list[int], y_axis: list[float], x_title: str, y_title: str, plot_title: str
) -> None:
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)
    plt.show()


# Function to create a scatter plot
def create_Scatter_Plot(
    x_axis: list[int], y_axis: list[float], x_title: str, y_title: str, plot_title: str
) -> None:
    plt.scatter(x_axis, y_axis)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)
    plt.show()


# Main function to execute the program
def main() -> None:
    mse_list = []
    r2_list = []
    ARGS = parse_Arguments()  # Parse command-line arguments
    DATASET = load_Dataset(ARGS.file)  # Load the dataset
    BATCH_SIZE = ARGS.batch
    TRAINING_SPLIT_PERCENT = ARGS.percent
    TRAINING_SPLIT_SCHOOL = ARGS.school
    TRAINING_SPLIT_YEARS = [ARGS.years[0], ARGS.years[1]]
    SAVE = ARGS.save
    TESTFILE = ARGS.test
    TRAINFILE = ARGS.train

    # Get training and test datasets
    TRAINING_FRAME, TEST_FRAME = get_Datasets(
        dataset=DATASET,
        trainingPercent=TRAINING_SPLIT_PERCENT,
        trainingSchool=TRAINING_SPLIT_SCHOOL,
        trainingYears=TRAINING_SPLIT_YEARS,
        save=SAVE,
        trainFile=TRAINFILE,
        testFile=TESTFILE,
    )

    EPOCHS = ARGS.epochs
    LEARNING_RATE = ARGS.learn
    ENABLE_PLOT = ARGS.plot

    # Create DataLoader for training and testing
    train_Loader = create_Datasets(dataset=TRAINING_FRAME, batch_size=BATCH_SIZE)
    test_Loader = create_Datasets(dataset=TEST_FRAME, batch_size=BATCH_SIZE)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Check for CUDA availability
    torch.manual_seed(0)  # Set a random seed for reproducibility

    # Define model architecture
    MODEL = nn.Sequential(
        nn.Linear(12, BATCH_SIZE),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE, BATCH_SIZE),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE, BATCH_SIZE // 2),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE // 2, BATCH_SIZE // 4),
        nn.ReLU(),
        nn.Linear(BATCH_SIZE // 4, 1),  # Output layer
    ).to(device)

    loss_function = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.SGD(
        MODEL.parameters(), lr=LEARNING_RATE
    )  # Stochastic gradient descent optimizer

    # Simulate training for multiple epochs
    loss_list = []
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n")
        loss_list += train(
            dataloader=train_Loader,
            model=MODEL,
            loss_fn=loss_function,
            optimizer=optimizer,
            device=device,
        )
        mse, r2 = test(
            dataloader=test_Loader, model=MODEL, loss_fn=loss_function, device=device
        )
        mse_list.append(mse)
        r2_list.append(r2)

    print("Finished!")

    # Generate plots if required
    if ENABLE_PLOT:
        create_Plot(
            x_axis=list(range(EPOCHS)),
            y_axis=r2_list,
            x_title="Epoch",
            y_title="R2",
            plot_title="R2 Over Time",
        )
        create_Plot(
            x_axis=list(range(EPOCHS)),
            y_axis=mse_list,
            x_title="Epoch",
            y_title="MSE",
            plot_title="MSE Over Time",
        )

        create_Scatter_Plot(
            x_axis=list(range(EPOCHS)),
            y_axis=mse_list,
            x_title="Epoch",
            y_title="Loss",
            plot_title="Loss Over Time",
        )


if __name__ == "__main__":
    main()  # Execute the main function
