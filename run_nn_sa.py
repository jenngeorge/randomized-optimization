import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def sa(X, y, lr, max_iters, decay, schedule, title, filename):
    nn = algos.NN(
        X, y,
        hidden_nodes = [4], 
        activation = 'relu',
        algorithm = 'simulated_annealing', 
        max_iters = max_iters,
        learning_rate = lr,
        schedule=schedule,
        clip_max = 50
    )
    train_acc, train_f1, test_acc, test_f1, log_loss_curves, loss = nn.run(10)
    # save curves
    fn = "results/{}.csv".format(filename)
    np.savetxt(fn, log_loss_curves, delimiter=",")
    plot_helper.plot_curves(log_loss_curves, title, x_label="Iterations", y_label="log loss", filename=filename)
    notes = """
        1 layer, [4] hidden nodes  
        clip max  = 50 
        activation = relu 
        max iters = {max_iters}
        learning rate = {lr}
        GeomDecay decay = {d}
    """.format(max_iters=max_iters, lr=lr, d=decay)
    model_helper.save_nn_report(title, filename, 
        train_acc, train_f1, test_acc, test_f1, loss, notes)

def sa_tuning(X, y, lr=0.0001, max_iters=1000):
    decays = [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
    for d in decays:
        d_str = str(d).split(".")[1]
        lr_str = str(lr).split(".")[1]
        schedule = mlrose.GeomDecay(init_temp=1, decay=d, min_temp=0.001)
        sa(X, y, 
            lr=lr, max_iters=max_iters, 
            decay=d,
            schedule=schedule,
            title="NN SA decay=0.{d} lr=0.{lr}".format(d=d_str, lr=lr_str), 
            filename="nn/sa/lr{lr}/d{d}lr{lr}".format(d=d_str, lr=lr_str))
    
if __name__ == "__main__":
    X, x_names, y = data_helper.get_mushroom_data("odor")
    # try with backprop lr
    sa_tuning(X, y, lr=0.0001,  max_iters=5000)
    
    # try with faster learning rate 
    sa_tuning(X, y, lr=0.1, max_iters=10000)
    
    # try best decay = 0.95 with more iterations 
    d = 0.95
    schedule = mlrose.GeomDecay(init_temp=1, decay=d, min_temp=0.001)
    lr = 0.1
    d_str = str(d).split(".")[1]
    lr_str = str(lr).split(".")[1]
    sa(X, y, 
        lr=lr, max_iters=20000, 
        decay=d,
        schedule=schedule,
        title="NN SA decay=0.{d} lr=0.{lr}".format(d=d_str, lr=lr_str), 
        filename="nn/sa/extra_runs/d{d}lr{lr}".format(d=d_str, lr=lr_str))
    
    
    # try much lower decay = 0.5
    d = 0.5
    schedule = mlrose.GeomDecay(init_temp=1, decay=d, min_temp=0.001)
    lr = 0.1
    d_str = str(d).split(".")[1]
    lr_str = str(lr).split(".")[1]
    sa(X, y, 
        lr=lr, max_iters=10000, 
        decay=d,
        schedule=schedule,
        title="NN SA decay=0.{d} lr=0.{lr}".format(d=d_str, lr=lr_str), 
        filename="nn/sa/extra_runs/d{d}lr{lr}".format(d=d_str, lr=lr_str))
    