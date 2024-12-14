# COS_470_Homework_5
### Name: 
Jacob Lorenzo
### Assignment:
Homework 5
### Class: 
COS 470/570
### Instructor: 
Dr. Hutchinson
### Date: 
11/22/24 - 12/13/24

#### Library Requirements:
- Check requirements.txt

## Write-Up

I have implemented several methods for generating and partitioning the dataset. The arguments in my code allow the dataset to be divided in different ways: by specific years, a fixed percentage of the dataset, and a percentage of schools. This flexibility enables the creation of various training and testing sets, which could be useful for different experimental setups.

Regarding the model architecture, I used five linear layers in my network with the following configuration: 12 → 17 → 17 → 8 → 2 → 1. I selected the ReLU activation function as it is widely used in such regression tasks. While I considered using Leaky ReLU as an alternative, I opted to stick with ReLU for simplicity and familiarity, as it is effective for many tasks.

For hyperparameters, I selected a batch size of 17, 100 epochs, and a learning rate of 0.00001. After discussing with my peers, it became clear that the dataset is fairly resilient to small changes, and adjusting the hyperparameters didn’t result in significant changes in the R² value. This is reflected in the graphs, which show that the R² values stabilize after 40 epochs (see Figures 1, 2, and 3 for MSE, R², and Loss, respectively). These graphs indicate that the model achieves a certain point of convergence, where further changes to the parameters did not improve performance significantly.

Overall, I found the assignment quite enjoyable. It took some time to properly set up the network and get everything working smoothly, but once that was done, training and evaluation proceeded without many issues. The most interesting part of the process was observing how the MSE and Loss follow identical trends in the graphs, with R² following the opposite trend. This is expected in regression tasks, as the MSE reflects the error magnitude, while R² measures the proportion of variance explained by the model.

## Howto
### Quickstart Commands: 
* I use Python 3.12, but it will depend on what version you use:

#### Command I Used for Results: ```py -3.12 main.py -pl -b 17 -tr training_school.csv -te testing_school.csv -e 100 -l .00001```

#### Using Years: ```py -3.12 main.py -pl -b 32 -e 200```

#### Using Percent of Dataframe: ```py -3.12 main.py -pl -b 32 -e 200 -p .15```

#### Using Percent of Schools: ```py -3.12 main.py -pl -b 32 -e 200 -s .1```

#### Using CSV Files: ```py -3.12 main.py -pl -b 32 -tr training_x.csv -te testing_y.csv```

    -  Replace testing_x and training_y with the appropriate files to replicate the above functionality. 

* You can add a -h or --help argument to either command to see the arguments in detail and the default values.

* I chose to set up my script this way because it allows for customizability. I'm unfamiliar with machine learning, and it's always best to parameterize and use arguments when possible. 