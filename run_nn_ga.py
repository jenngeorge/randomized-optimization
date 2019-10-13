import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def ga(X, y, lr, title, filename, schedule=None, max_iters=1000, pop_size=200, mutation_prob=0.1):
    nn = algos.NN(
        X, y,
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'genetic_alg', 
        schedule=schedule,
        max_iters = max_iters,
        learning_rate = lr,
        clip_max = 50, 
        max_attempts = 1000,
        pop_size = pop_size,
        mutation_prob = mutation_prob
    )
    train_acc, test_acc, log_loss_curves, fitted_weights, loss = nn.run(10)
    # save curves
    fn = "results/{}.csv".format(filename)
    np.savetxt(fn, log_loss_curves, delimiter=",")
    plot_helper.plot_curves(log_loss_curves, title, x_label="Iterations", y_label="log loss", filename=filename, y_range=[-0.1, 1.1])
    notes = """
        1 layer, 4 hidden nodes  
        clip max  = 50 
        activation = relu 
        max iters = {max_iters}
        learning rate = {lr}
        pop_size = {pop_size}
        mutation_prob = {mutation_prob}
    """.format(max_iters=max_iters, lr=lr, pop_size=pop_size, m_prob=mutation_prob)
    model_helper.save_nn_report(title, filename, train_acc, test_acc, fitted_weights, loss, notes)


def ga_tuning(X, y, schedule=None, max_iters=1000):
    lrs = [0.0001, 0.001, 0.01]
    # remember weights len = 196 so this is like 
    # wl, wl*5, wl*10, wl*15
    pop_size = [200, 1000, 2000, 3000]
    mutation_probs = [0.1, 0.01, 0.001]
    for lr in lrs:
        for p in pop_size: 
            for m in mutation_probs:
                lr_str = str(lr).split(".")[1]
                m_str = str(m).split(".")[1]
                ga(X, y, 
                    lr=lr, max_iters=max_iters, 
                    schedule=schedule,
                    pop_size =  p,
                    mutation_prob = m,
                    title="NN GA lr=0.{lr} m_prob={m} pop_size={p}".format(lr=lr_str, m=m_str, p=p), 
                    filename="nn/ga/lr{lr}m{m}p{p}".format(lr=lr_str, m=m_str, p=p)
                    )
                    
                    
if __name__ == "__main__":
    X, x_names, y = data_helper.get_mushroom_data("odor")
    ga_tuning(X, y, schedule=mlrose.GeomDecay(decay=0.9), max_iters=500)
    # started at 5pm


    


    