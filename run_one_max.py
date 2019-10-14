import mlrose
import numpy as np

from helpers import algos, model_helper, run_opt


if __name__ == "__main__":
    fitness = mlrose.OneMax()

    # tune sa with n=10 (not in mimic paper)
    problem = mlrose.DiscreteOpt(length = 10, fitness_fn = fitness, maximize = True, max_val = 2)
    run_opt.tune_sa(problem, max_iters=np.inf, filedir="one_max/sa/tuning/")
    

    sizes = [10, 20, 30, 40, 50, 60]
    for s in sizes:
        problem = mlrose.DiscreteOpt(length = s, fitness_fn = fitness, maximize = True, max_val = 2)
        file_dir = "one_max/n={}/".format(s)
    
        run_opt.run_rhc(problem, max_attempts=10, max_iters=np.inf, restarts=0, n=10, filedir=file_dir)
    
        schedule = mlrose.GeomDecay(init_temp=1, decay=0.95, min_temp=0.001)
        run_opt.run_sa(problem, decay=0.95, schedule=schedule, max_iters=np.inf, filedir=file_dir)
    
        run_opt.run_ga(problem, m_prob=0.1, pop_size=200, max_iters=np.inf, filedir=file_dir)
    
        run_opt.run_mimic(problem, keep_pct=0.2, pop_size=200, max_iters=np.inf, filedir=file_dir)
    