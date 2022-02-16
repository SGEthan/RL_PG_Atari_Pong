# Training & Playing Atari-Pong with Policy Gradient Algorithm

Here we give our final project for our Machine Learning course. We trained our model with the **Policy Gradient Algorithm**.

For the full report, see `Report` directory

### Files

Here's the tree of our project directory:

```
.
├── Reward_record.csv
├── Time_record.csv
├── checkpoints
├── func.py
├── main.py
├── my_model.py
├── plots.py
├── readme.md
├── Final_model.p
├── reward.jpg
├── score.jpg
├── time.jpg
├── Report
│   ├── Report.pdf
│   └── Report.md
```

Here's introduction of each folder and file:

* `Reward_record.csv`: This csv file record the reward of every episode when we train our model.
* `Time_record.csv`: This csv file record the time wasted of  every episode when we train our model.
* `checkpoints`: This folder is used to store our checkpoints when we train our model.
* `func.py`: Here we give the definition of some functions we used in our algorithm.
* `main.py`: This is the main script to run when we train or test a model.
* `my_model.py`: Here we give our definition of our agent, used to call in `main.py`.
* `plots.py`: This is the script to visualize the reward record and time record.
* `Final_model.p`: This is our final trained model to show.

### How to run

First we run `python main.py`, and we can get the instruction:

```
$ python main.py

choose to load checkpoints or initialize one:
1.choose one
2.build one randomly
```

so we can load a trained model or build one from scratch. When we type `1` we get:

```
$ 1
enter the ckpt file name:
```

so now we can load our model from a checkpoints by typing the name of the checkpoint file, such as our `Final_model.p`.

After we initialized our model, we get the instruction:

```
choose to do:
1.train 2.test
```

so now we can store  checkpoints every 100 episodes , successfully run the code! 