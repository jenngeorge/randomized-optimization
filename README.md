This repo uses https://github.com/gkhayes/mlrose
& some helper code from my first project https://github.com/jenngeorge/supervised-learning

## setup 
Ensure you have Python3 and [conda](https://docs.conda.io/en/latest/) installed.

With conda, create and activate the environment from environment.yml like `conda env create -f environment.yml`

Follow the prompt to activate the environment with `conda activate ro_env`

## running optimization problem experiments 
run the corresponding file in this directory in the `ro_env` environment like `python3 run_one_max.py`
a txt file containing info and performance metrics for each run will be written to `results/{problem abbreviation}`


## plotting optimization problem experiments 
I copied metrics from each experiment's txt file to `make_opt_plots.py`

If you wish to recreate the plots for your own experiment runs, 
you will need to do the same. 


## running neural network experiments 
run the corresponding file in this directory in the `ro_env` environment 
I recommend commenting out portions of `__main__` for faster runtime 

backprop: `python3 run_nn_backprop.py`
rhc: `python3 run_nn_rhc.py`
simulated annealing: `python3 run_nn_sa.py`
genetic algorithm: `python3 run_nn_ga.py`


a csv file containing the learning curve, 
txt file containing stats and info, and a plot image 
will be written to `results/nn` for each experiment run in each file. 
