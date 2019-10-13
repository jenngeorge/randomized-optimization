import numpy as np
import json
from sklearn import model_selection, metrics

from helpers import data_helper, plot_helper

def save_nn_report(title, filename, train_acc, train_f1, test_acc, test_f1, loss, notes=""):
    train_acc_mean = "train acc mean: {} \n\n".format(np.mean(train_acc))
    train_acc_std = "train acc std: {} \n\n".format(np.std(train_acc))
    test_acc_mean = "test acc mean: {} \n\n".format(np.mean(test_acc))
    test_acc_std = "test acc std: {} \n\n".format(np.std(test_acc))
    
    train_f1_mean = "train f1 mean: {} \n\n".format(np.mean(train_f1))
    train_f1_std = "train f1 std: {} \n\n".format(np.std(train_f1))
    test_f1_mean = "test f1 mean: {} \n\n".format(np.mean(test_f1))
    test_f1_std = "test f1 std: {} \n\n".format(np.std(test_f1))
        
    loss = "loss: {}".format(", ".join([str(l) for l in loss]))
    fn = "results/{}.txt".format(filename)
    with open(fn, "w+") as fh:
        fh.write(title)
        fh.write("\n")
        fh.write(notes)
        fh.write("\n")
        fh.write("train accuracy: ")
        fh.write(", ".join([str(a) for a in train_acc]))
        fh.write("\n\n")
        fh.write("train f1: ")
        fh.write(", ".join([str(a) for a in train_f1]))
        fh.write("\n")
        fh.write(train_acc_mean)
        fh.write(train_acc_std)
        fh.write(train_f1_mean)
        fh.write(train_f1_std)
        fh.write("test accuracy: ")
        fh.write(", ".join([str(a) for a in test_acc]))
        fh.write("\n\n")
        fh.write("test f1: ")
        fh.write(", ".join([str(a) for a in test_f1]))
        fh.write("\n")
        fh.write(test_acc_mean)
        fh.write(test_acc_std)
        fh.write(test_f1_mean)
        fh.write(test_f1_std)
        fh.write(loss)
        
    print("saved ", fn)
    