import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def rhc(X, y, lr, title, filename, schedule=None, max_iters=1000):
    nn = algos.NN(
        X, y,
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'random_hill_climb', 
        schedule=schedule,
        max_iters = max_iters,
        learning_rate = lr,
        clip_max = 50, 
        max_attempts = 1000
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
    """.format(max_iters=max_iters, lr=lr)
    model_helper.save_nn_report(title, filename, train_acc, test_acc, 
        fitted_weights, loss, notes)

def rhc_tuning(X, y, schedule=None):
    lrs = [0.0001, 0.001, 0.01]
    for lr in lrs:
        lr_str = str(lr).split(".")[1]
        rhc(X, y, 
            lr=lr, max_iters=1000, 
            schedule=schedule,
            title="NN RHC lr=0.{lr} decay={d}".format(lr=lr_str, d=".9" if schedule else ".99"), 
            filename="nn/rhc/lr{lr}{d}".format(lr=lr_str, d="d9" if schedule else ""))
    
    
if __name__ == "__main__":
    X, x_names, y = data_helper.get_mushroom_data("odor")
    rhc_tuning(X, y)
    rhc_tuning(X, y, schedule=mlrose.GeomDecay(decay=0.9))
    # try different schedule 


    


    