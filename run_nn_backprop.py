import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def backprop(X, y, hn, lr, max_iters, title, filename):
    nn = algos.NN(
        X, y,
        hidden_nodes = hn, 
        activation = 'relu',
        algorithm = 'gradient_descent', 
        max_iters = max_iters,
        learning_rate = lr,
        clip_max = 50
    )
    train_acc, train_f1, test_acc, test_f1, log_loss_curves, loss = nn.run(10)
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
    model_helper.save_nn_report(title, filename, 
        train_acc, train_f1, test_acc, test_f1, loss, notes)

def backprop_tuning(X, y, hn):
    lrs = [0.00005, 0.0001, 0.0005, 0.001]
    for lr in lrs:
        lr_str = str(lr).replace(".", "-")
        backprop(X, y, 
            hn=hn,
            lr=lr, max_iters=2000, 
            title="NN backprop lr={lr}, hidden nodes={hn}".format(lr=lr_str, hn=hn), 
            filename="nn/backprop/hn{hn}lr{lr}".format(lr=lr_str, hn=hn))
    
if __name__ == "__main__":
    X, x_names, y = data_helper.get_mushroom_data("odor")
    # try 2 hidden nodes 
    backprop_tuning(X, y, [2])
    # # try 4 hidden nodes 
    backprop_tuning(X, y, [4])
    
    