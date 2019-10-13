import matplotlib.pyplot as plt
import numpy as np


def plot_curves(curves, title, x_label="Iterations", y_label="Fitness", x_range=None, filename=None, is_log=False, y_range=None, y_ticks=None):
    """
    adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    and https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    """
    if not x_range:
        x_range = np.arange(curves.shape[1])
    plt.figure()
    plt.title(title)
    if y_range:
        plt.ylim(y_range[0], y_range[1])
        plt.yticks(y_ticks)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    curves_mean = np.mean(curves, axis=0)
    curves_std = np.std(curves, axis=0)

    plt.grid()

    if is_log:
        plt.semilogx(x_range, curves_mean,
                     color="navy", lw=1, markersize=2, basex=2)
    else:
        plt.plot(x_range, curves_mean, 'o-', color="r",
                 lw=1, markersize=2)

    plt.fill_between(x_range, curves_mean - curves_std,
                     curves_mean + curves_std, alpha=0.1,
                     color="r")

    # write plot
    fn = filename if filename else lower(title).replace(" ", "_")
    path = "results/{fn}".format(fn=fn)
    plt.savefig(path)
    print("saved ", path)


def plot_n_curves(arr_dict, title, filename,  x_label="Iterations", y_label="Score"):
    """
    given a dict of arrays, plot each array 
    x = length of longest array 
    """
    colors = ["mediumblue", "darkorange", "red", "green", "darkviolet"]
    plt.figure()
    plt.title(title)
    plt.ylim(0.0, 1.1)
    plt.yticks(np.arange(0.0, 1.1, 0.05))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()
    
    max_x_range = 0 
    for k in arr_dict.keys():
        l = len(arr_dict[k])
        if l >= max_x_range:
            max_x_range = l
            
    # x_range = np.arange(0, x_range)
    # plt.plot(np.arange(0, len(arr_dict[1])), arr_dict[1], 'o-', color=colors[1],
    #          label=1, lw=1, markersize=2)
    for k in arr_dict.keys():
        plt.plot(np.arange(0, len(arr_dict[k])), arr_dict[k], 'o-', color=colors[k],
                 label=k, lw=1, markersize=2)
    
    # write plot
    fn = filename if filename else lower(title).replace(" ", "_")
    path = "results/{fn}".format(fn=fn)
    plt.savefig(path)
    print("saved ", path)