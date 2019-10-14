import numpy as np 


def plot_opt(arr_dict, title, filename,  x_label="Inputs", y_label="Evaluations"):
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
    
    x_range = [10, 20, 30, 40, 50, 60]
            
    for k in arr_dict.keys():
        plt.plot(np.arange(0, len(arr_dict[k])), arr_dict[k], 'o-', color=colors[k],
                 label=k, lw=1, markersize=2)
    
    # write plot
    fn = filename if filename else lower(title).replace(" ", "_")
    path = "opt_plots/{fn}.png".format(fn=fn)
    plt.savefig(path)
    print("saved ", path)

    


if __name__ == "__main__":
    # one max 
    om_dict = {
        "mimic": [12.4],
        "ga": [],
        "rhc": [],
        "sa": []
    }
    
    om_dict = {
        "mimic": [0.9165151389911681],
        "ga"
        "rhc": [],
        "sa": []
    }
    
    plot_opt(om_dict,
        title="Function Evals Required to Maximize OneMax",
        filename="onemax")
    
    
    # four peaks 
    
    # k colors 
    
    # continuos peaks 