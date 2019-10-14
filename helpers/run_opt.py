import mlrose
import numpy as np

from helpers import algos, model_helper

# helper methods for running & tuning algos for opt problems 

def run_rhc(problem, max_attempts=10, max_iters=1000, restarts=0, n=10, filedir=""):
    rhc = algos.RHC(problem, max_attempts=max_attempts, max_iters=max_iters, restarts=restarts)
    best_fitnesses, learning_curves = rhc.run(n)
    note = """
        max iters = {max_iters}
        max_attempts = {ma}
        restarts = {r}
    """.format(max_iters=max_iters, ma=max_attempts, r=restarts)
    best_fitnesses, learning_curves = rhc.run(n)
    title = "RHC restarts={r}".format(ma=max_attempts, r=restarts)
    filename = "{fd}RHCma{ma}r{r}".format(fd=filedir, ma=max_attempts, r=restarts)
    model_helper.save_algo_runs(title, filename, learning_curves, best_fitnesses, note)
    
###
    
def run_sa(problem, schedule, decay, max_attempts=10, max_iters=1000, n=10, filedir=""):
    sa = algos.SA(problem, schedule = schedule, max_attempts = max_attempts, max_iters = max_iters)
    best_fitnesses, learning_curves = sa.run(n)
    note = """
        max iters = {max_iters}
        max_attempts = {ma}
        schedule = GeomDecay with {d} decay
    """.format(max_iters=max_iters, ma=max_attempts, d=decay)
    d_str = str(decay).split(".")[1]
    title = "SA decay=0.{d}".format(d=d_str)
    filename = "{fd}SAd{d}".format(fd=filedir, d=d_str)
    model_helper.save_algo_runs(title, filename, learning_curves, best_fitnesses, note)
    
def tune_sa(problem, max_attempts=10, max_iters=np.inf, n=10, filedir=""):
    decays = [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
    for d in decays:
        schedule = mlrose.GeomDecay(init_temp=1, decay=d, min_temp=0.001)
        run_sa(problem, schedule, d, max_attempts, max_iters, n, filedir=filedir)
    
###
    
def run_ga(problem, m_prob, pop_size, max_attempts=10, max_iters=1000, n=10, filedir=""):
    ga = algos.GA(problem, mutation_prob=m_prob, pop_size=pop_size, max_attempts = max_attempts, max_iters = max_iters)
    best_fitnesses, learning_curves = ga.run(n)
    note = """
        max iters = {max_iters}
        max_attempts = {ma}
        pop size = {pop_size}
        mutation prob = {m_prob}
    """.format(max_iters=max_iters, ma=max_attempts, pop_size=pop_size, m_prob=m_prob)
    m_str = str(m_prob).split(".")[1]
    title = "GA m_prob=0.{m} pop={p}".format(m=m_str, p=pop_size)
    filename = "{fd}GAm{m}p{p}".format(fd=filedir, m=m_str, p=pop_size)
    model_helper.save_algo_runs(title, filename, learning_curves, best_fitnesses, note)
    
def tune_ga(problem, state_length=1, max_attempts=10, max_iters=1000, n=10, filedir=""):
    pop_sizes = [0.1, 1, 5, 10]
    m_prob = [0.1, 0.01, 0.001]
    for p in pop_sizes:
        size = int(p*state_length)
        p_size = size if size > 0 else 1
        
        for m in m_prob:
            run_ga(problem, m_prob=m, pop_size=p_size, max_attempts=10, max_iters=1000, n=10, filedir=filedir)
    
###

def run_mimic(problem, keep_pct, pop_size, max_attempts=10, max_iters=1000, n=10, filedir=""):
    ga = algos.MIMIC(problem, keep_pct=keep_pct, pop_size=pop_size, max_attempts = max_attempts, max_iters = max_iters)
    best_fitnesses, learning_curves = ga.run(n)
    note = """
        max iters = {max_iters}
        max_attempts = {ma}
        pop size = {pop_size}
        keep_pct = {keep_pct}
    """.format(max_iters=max_iters, ma=max_attempts, pop_size=pop_size, keep_pct=keep_pct)
    kp_str = str(keep_pct).split(".")[1]
    title = "MIMIC keep_pct=0.{kp} pop={p}".format(kp=kp_str, p=pop_size)
    filename = "{fd}MIMICp{p}kp{kp}".format(fd=filedir, kp=kp_str, p=pop_size)
    model_helper.save_algo_runs(title, filename, learning_curves, best_fitnesses, note)
    

    
    