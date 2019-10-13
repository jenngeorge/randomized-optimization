import numpy as np
import json
from sklearn import model_selection, metrics

from helpers import data_helper, plot_helper

def save_nn_report(title, filename, train_acc, test_acc, weights, loss, notes=""):
    train_acc_mean = "train acc mean: {}".format(np.mean(train_acc))
    train_acc_std = "train acc std: {}".format(np.std(train_acc))
    test_acc_mean = "test acc mean: {}".format(np.mean(test_acc))
    test_acc_std = "test acc std: {}".format(np.std(test_acc))
    fitted_weights = "fitted weights: \n {}".format("\n".join([np.array2string(w, separator=',') for w in weights]))
    loss = "loss: {}".format(", ".join([str(l) for l in loss]))
    fn = "results/{}.txt".format(filename)
    with open(fn, "w+") as fh:
        fh.write(title)
        fh.write("\n")
        fh.write(notes)
        fh.write("\n")
        fh.write("train accuracy: ")
        fh.write(", ".join([str(a) for a in train_acc]))
        fh.write("\n")
        fh.write(train_acc_mean)
        fh.write("\n")
        fh.write(train_acc_std)
        fh.write("\n")
        fh.write("test accuracy: ")
        fh.write(", ".join([str(a) for a in test_acc]))
        fh.write("\n")
        fh.write(test_acc_mean)
        fh.write("\n")
        fh.write(test_acc_std)
        fh.write("\n")
        fh.write(loss)
        fh.write("\n")
        fh.write(fitted_weights)
        
    print("saved ", fn)
    
def test_report(clf, X, y, test_size, r_seed):
    """
    elements of the score: 
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
    """
    X_train, X_test, y_train, y_test = data_helper.train_test_split(X, y, r_seed, test_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_pred, digits=5)
    return report 
    

def save_test_report(clf_name, data_name, clf, X, y, test_size, r_seed, note=""):
    head = "{data_name}: {clf_name}\n\n".format(data_name=data_name, clf_name=clf_name)
    report = test_report(clf, X, y, test_size, r_seed)
    
    param_str = ""
    param_dict = clf.get_params()
    for k in param_dict:
        param_str += "{key}: {val}\n".format(key=k, val=param_dict[k])
        
    fn = "reports/{data_name}/{clf_name}{note}.txt".format(
        data_name=data_name.lower(),
        clf_name=clf_name.lower(),
        note=note
    )
    # save to file 
    with open(fn, "w+") as fh:
        fh.write(head)
        fh.write(report)
        fh.write("\n")
        fh.write("clf params: \n")
        fh.write(param_str)