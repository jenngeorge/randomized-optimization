import mlrose
import numpy as np

import algos 

def run_rhc(problem, init_state, n=1):
    rhc = algos.RHC(problem, max_attempts=10, max_iters=np.inf, init_state=init_state)
    best_fitnesses, learning_curves = rhc.run(n)
    return best_fitnesses, learning_curves
    
def run_sa(problem, init_state, n=1):
    # Define decay schedule
    # TODO: tune 
    # https://github.com/gkhayes/mlrose/blob/master/mlrose/decay.py#L157
    schedule = mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
    
    sa = algos.SA(problem, schedule = schedule, max_attempts = 10, max_iters = np.inf, init_state = init_state)
    best_fitnesses, learning_curves = sa.run(n)
    return best_fitnesses, learning_curves
    
def run_ga(problem, init_state, n=1):
    ga = algos.GA(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    best_fitnesses, learning_curves = ga.run(n)
    return best_fitnesses, learning_curves

def run_mimic(problem, init_state, n=1):
    mimic = algos.MIMIC(problem, pop_size=200, keep_pct=0.1, max_attempts=10,
          max_iters=np.inf)
    best_fitnesses, learning_curves = mimic.run(n)
    return best_fitnesses, learning_curves    


if __name__ == "__main__":
    runs = 5
    
    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(length = 20, fitness_fn = fitness, maximize = True, max_val = 2)
    init_state=np.zeros(20)
    
    rhc_best_fitnesses, rhc_learning_curves = run_rhc(problem, init_state, n=runs)
    print('rhc')
    print(rhc_best_fitnesses)
    # print(rhc_learning_curves)
    
    sa_best_fitnesses, sa_learning_curves = run_sa(problem, init_state, n=runs)
    print('sa')
    print(sa_best_fitnesses)
    # print(sa_learning_curves)
    
    ga_best_fitnesses, ga_learning_curves = run_ga(problem, init_state, n=runs)
    print('ga')
    print(ga_best_fitnesses)
    # print(ga_learning_curves)
    
    mimic_best_fitnesses, mimic_learning_curves = run_mimic(problem, init_state, n=runs)
    print('mimic')
    print(mimic_best_fitnesses)
    # print(mimic_learning_curves)
    