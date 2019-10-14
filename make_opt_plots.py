import numpy as np 
import matplotlib.pyplot as plt


def plot_opt(arr_dict, title, filename,  x_label="Inputs", y_label="Iterations"):
    """
    given a dict of arrays, plot each array 
    x = length of longest array 
    """
    colors = {
        "ga": "mediumblue",
        "mimic": "darkorange",
        "rhc": "green",
        "sa": "darkviolet"
    }
    # }"mediumblue", "darkorange", "red", "green", "darkviolet"]
    plt.figure()
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim((0, 100))

    plt.grid()
    
    x_range = [10, 20, 30, 40, 50, 60]
            
    for k in arr_dict.keys():
        plt.plot(x_range, arr_dict[k], 'o-', color=colors[k],
                 label=k, lw=1, markersize=2)
    plt.legend()
    # write plot
    fn = filename if filename else lower(title).replace(" ", "_")
    path = "opt_plots/{fn}.png".format(fn=fn)
    plt.savefig(path)
    print("saved ", path)

    


if __name__ == "__main__":
    # one max 
    om_iters_dict = {
        "ga": [12.4, 17.0, 21.2, 19.3, 18.1, 19.9],
        "mimic": [11.2, 13.0, 14.2, 15.1, 16.6, 17.5],
        "rhc": [22.7, 30.6, 38.6, 46.8, 70.8, 64.8],
        "sa": [33.1, 37.5, 51.0, 49.6, 68.5, 68.5]
    }
    om_fit_dict = {
        "ga": [10.0,  18.2, 25.3, 32.3, 39.0, 44.8],
        "mimic": [10.0, 20.0, 30.0, 40.0, 50.0, 59.9],
        "rhc": [9.6, 16.4, 25.9, 32.8, 42.3, 49.2],
        "sa": [9.7, 17.1, 26.3, 32.4, 42.8, 49.5]
    }
    plot_opt(om_iters_dict,
        title="Mean Iterations Required to Maximize OneMax",
        filename="onemax_iters")
    plot_opt(om_fit_dict,
        title="Mean Fitness Achieved for OneMax",
        filename="onemax_fit", y_label="Fitness")    
    
    
    # cont peaks 
    contpeaks_iters_dict = {
        "ga": [11.1, 16.4, 20.9, 21.9, 23.2, 14.2,],
        "mimic": [11.2, 14.3, 15.5, 18.1, 20.2, 19.2],
        "rhc": [20.4, 19.8, 22.3, 15.8, 14.8, 15.3],
        "sa": [35.8, 72.0, 88.7, 228.5, 531.5, 556.6]
    }
    contpeaks_fit_dict = {
        "ga": [17.0, 33.0,  47.4, 59.4, 69.9, 78.1],
        "mimic": [17.0, 33.7, 49.5, 63.6, 75.3, 85.7],
        "rhc": [12.4, 22.5, 18.4, 11.9, 7.4, 7.6],
        "sa": [15.2, 34.1, 41.9, 56.5, 65.8, 63.3]
    }
    plot_opt(contpeaks_iters_dict,
        title="Mean Iterations Required to Maximize ContPeaks",
        filename="contpeak_iters")
    plot_opt(contpeaks_fit_dict,
        title="Mean Fitness Achieved for ContPeaks",
        filename="contpeak_fit", y_label="Fitness")   
    
    # four peaks 
    four_peaks_iters_dict = {
        "ga": [11.1, 17.7, 23.8, 23.6, 34.6, 22.6],
        "mimic": [11.5, 14.0, 17.4, 17.3, 17.0, 19.9],
        "rhc": [20.4, 13.8, 13.2, 12.8, 11.9, 11.6],
        "sa": [39.9, 111.5, 93.7, 331.6, 814.5, 1067.6]
    }
    four_peaks_fit_dict = {
        "ga": [17.0, 34.7, 46.0, 56.6, 68.3, 69.1],
        "mimic": [17.0, 34.9, 45.1, 53.0, 56.6, 67.0],
        "rhc": [7., 8.3,  3.6, 3.0, 1.5, 2.2],
        "sa": [13.2,  27.1, 31.8, 54.3, 53.1, 44.5]
    }
    plot_opt(four_peaks_iters_dict,
        title="Mean Iterations Required to Maximize FourPeaks",
        filename="four_peak_iters")
    plot_opt(four_peaks_fit_dict,
        title="Mean Fitness Achieved for FourPeaks",
        filename="four_peak_fit", y_label="Fitness")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # continuos peaks 