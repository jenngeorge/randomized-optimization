import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def backprop(X, y, hn, lr, max_iters, title, filename, schedule=None):
    nn = algos.NN(
        X, y,
        hidden_nodes = hn, 
        activation = 'relu',
        algorithm = 'gradient_descent', 
        max_iters = max_iters,
        learning_rate = lr,
        clip_max = 50, 
        max_attempts = 1000,
        schedule=schedule
    )
    train_acc, test_acc, log_loss_curves, fitted_weights, loss = nn.run(10)
    # save curves
    fn = "results/{}.csv".format(filename)
    np.savetxt(fn, log_loss_curves, delimiter=",")
    plot_helper.plot_curves(log_loss_curves, title, x_label="Iterations", y_label="log loss", filename=filename, y_range=[-0.1, 1.1])
    notes = """
        1 layer, {hn} hidden nodes  
        clip max  = 50 
        activation = relu 
        max iters = {max_iters}
        learning rate = {lr}
    """.format(hn=hn, max_iters=max_iters, lr=lr)
    model_helper.save_nn_report(title, filename, train_acc, test_acc, 
        fitted_weights, loss, notes)

def backprop_tuning(X, y, hn, schedule=None):
    lrs = [0.0001, 0.0005, 0.001]
    for lr in lrs:
        lr_str = str(lr).split(".")[1]
        backprop(X, y, 
            hn=hn,
            schedule=schedule,
            lr=lr, max_iters=1000, 
            title="NN backprop lr=0.{lr}, hidden nodes={hn}".format(lr=lr_str, hn=hn), 
            filename="nn/backprop/hn{hn}lr{lr}{d}".format(lr=lr_str, hn=hn, d="d999" if schedule else ""))
    
if __name__ == "__main__":
    # X, x_names, y = data_helper.get_avo_data("AveragePrice")
    X, x_names, y = data_helper.get_mushroom_data("odor")
    # try 2 hidden nodes 
    backprop_tuning(X, y, [2])
    # # try 4 hidden nodes 
    backprop_tuning(X, y, [4])
    
    # going to use faster decay for faster learning
    schedule = mlrose.GeomDecay(decay=0.9) # default is 0.99
    backprop(X, y, 
        hn=[4],
        schedule=schedule,
        lr=0.0001, max_iters=1000, 
        title="NN backprop lr=0.{lr}".format(lr="0001", hn=[4]), 
        filename="nn/backprop/hn{hn}lr{lr}{d}".format(lr="0001", hn=[4], d="d9" if schedule else ""))

    