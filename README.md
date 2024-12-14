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

I have several methods for generating and dividing the data sets. I have arguments that allow the data set to be divided by years, a flat percent of the dataset, and a percent of schools. 

I assume there are some improvements that can be made to my architecture, but I tried out 5 linear layers where it went 12 -> 17, 17 -> 17, 17 -> 8, 8 -> 2, 2 -> 1. I used the ReLU activation function since the only other type I thought could be viable was the leaky_ReLU.

For my hyperparameters, I ended up with a batch size of 17, 100 epochs, and a learning rate of .00001. After talking to my peers, it seems the data set is quite resilient to modification, so it didn't do a lot to change the r^2 value as the attached graphs show that it flattens out (Figure_1_MSE.png, Figure_2_R2.png, Figure_3_Loss.png). 

Overall, the assignment was quite enjoyable. It took some time to get the network set up properly, but once it was set up, it was smooth sailing. 

I thought that it was quite interesting how the MSE and Loss both follow identical graphs, and then the R^2 follows the same graph but flipped.

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