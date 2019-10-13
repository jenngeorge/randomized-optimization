import mlrose
import numpy as np
from sklearn import metrics 
from helpers import data_helper

# provides objects around the mlrose algos 
class RHC:
    # random hill climb source: https://github.com/gkhayes/mlrose/blob/master/mlrose/algorithms.py#L114
    def __init__(self, problem, max_attempts, max_iters, init_state, restarts=0):
        self.problem = problem 
        self.max_attempts  = max_attempts
        self.max_iters = max_iters 
        self.restarts = restarts
        self.init_state = init_state
        
    def run(self, n=1):
        """
        n : the number of runs to do 
        returns best_fitnesses (list), learning_curves (list)
        """
        best_fitnesses = []
        learning_curves = []
        for i in np.arange(n):
            _, _, learning_curve = mlrose.random_hill_climb(
                self.problem, 
                max_attempts = self.max_attempts,
                max_iters = self.max_iters,
                restarts = self.restarts,
                init_state = self.init_state, 
                random_state = None, curve=True)
            best_fitness = np.max(learning_curve)
            best_fitnesses.append(best_fitness)
            learning_curves.append(learning_curve)
            
        return best_fitnesses, learning_curves


class SA:
    # simulated annealing source: https://github.com/gkhayes/mlrose/blob/master/mlrose/algorithms.py#L225
    def __init__(self, problem, schedule, max_attempts, max_iters, init_state):
        self.problem = problem 
        self.schedule = schedule 
        self.max_attempts = max_attempts
        self.max_iters = max_iters 
        self.init_state = init_state
        
    def run(self, n=1):
        """
        n : the number of runs to do 
        returns best_fitnesses (list), learning_curves (list)
        """
        best_fitnesses = []
        learning_curves = []
        for i in np.arange(n):
            _, _, learning_curve = mlrose.simulated_annealing(
                self.problem, 
                schedule = self.schedule,
                max_attempts = self.max_attempts, 
                max_iters = self.max_iters,
                init_state = self.init_state, 
                random_state = None, curve=True)
            best_fitness = np.max(learning_curve)
            best_fitnesses.append(best_fitness)
            learning_curves.append(learning_curve)
            
        return best_fitnesses, learning_curves
        

class GA:
    # genetic alg source: https://github.com/gkhayes/mlrose/blob/master/mlrose/algorithms.py#L334
    def __init__(self, problem, pop_size, mutation_prob, max_attempts, max_iters):
        self.problem = problem 
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.max_iters = max_iters 
        
    def run(self, n=1):
        """
        n : the number of runs to do 
        returns best_fitnesses (list), learning_curves (list)
        """
        best_fitnesses = []
        learning_curves = []
        for i in np.arange(n):
            _, _, learning_curve = mlrose.genetic_alg(
                self.problem, 
                pop_size = self.pop_size,
                mutation_prob = self.mutation_prob,
                max_attempts = self.max_attempts, 
                max_iters = self.max_iters,
                random_state = None, curve=True)
            best_fitness = np.max(learning_curve)
            best_fitnesses.append(best_fitness)
            learning_curves.append(learning_curve)
            
        return best_fitnesses, learning_curves
        

class MIMIC:
    # mimic source: https://github.com/gkhayes/mlrose/blob/master/mlrose/algorithms.py#L458
    def __init__(self, problem, pop_size, keep_pct, max_attempts, max_iters):
        self.problem = problem 
        self.pop_size = pop_size
        self.keep_pct = keep_pct 
        self.max_attempts = max_attempts
        self.max_iters = max_iters 
        
    def run(self, n=1):
        """
        n : the number of runs to do 
        returns best_fitnesses (list), learning_curves (list)
        """
        best_fitnesses = []
        learning_curves = []
        for i in np.arange(n):
            _, _, learning_curve = mlrose.mimic(
                self.problem, 
                pop_size = self.pop_size,
                keep_pct = self.keep_pct,
                max_attempts = self.max_attempts, 
                max_iters = self.max_iters,
                random_state = None, curve=True)
            best_fitness = np.max(learning_curve)
            best_fitnesses.append(best_fitness)
            learning_curves.append(learning_curve)
            
        return best_fitnesses, learning_curves
        
class NN:
    # nn source: https://github.com/gkhayes/mlrose/blob/master/mlrose/neural.py#L746
    # data = dict with keys "X_train", "y_train", "X_test", "y_test"
    def __init__(self,
        X,
        y,
        test_size=0.1,
        hidden_nodes=[2], 
        activation='relu', 
        algorithm='random_hill_climb',
        max_iters=100,
        is_classifier=True,
        learning_rate=0.1,
        early_stopping=True,
        clip_max=1e+10,
        restarts=0,
        schedule=mlrose.GeomDecay(),
        pop_size=200,
        mutation_prob=0.1,
        max_attempts=10
    ):
        self.X = X
        self.y = y 
        self.test_size = test_size
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.algorithm = algorithm
        self.max_iters = max_iters
        self.bias = True
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.restarts = restarts
        self.schedule = schedule if schedule else mlrose.GeomDecay()
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.random_state = None
        self.curve = True

    def run(self, n=1):
        """
        n : the number of runs to do 
        returns best_fitnesses (list), learning_curves (list)
        """
        log_loss_curves = []
        train_acc = []
        test_acc = []
        train_f1 = []
        test_f1 = []
        fitted_weights = []
        loss = []
        for i in np.arange(n):
            X_train, X_test, y_train, y_test = data_helper.train_test_split(self.X, self.y, seed=None, test_size=self.test_size)
            model = mlrose.NeuralNetwork(
                hidden_nodes=self.hidden_nodes, 
                activation=self.activation, 
                algorithm=self.algorithm,
                max_iters=self.max_iters,
                bias=True,
                is_classifier=True,
                learning_rate=self.learning_rate,
                early_stopping=False,
                clip_max=self.clip_max,
                restarts=self.restarts,
                schedule=self.schedule,
                pop_size=self.pop_size,
                mutation_prob=self.mutation_prob,
                max_attempts=self.max_attempts,
                curve=True
            )
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
            train_acc.append(y_train_accuracy)
            train_f1.append(metrics.f1_score(y_train, y_train_pred, average='weighted'))
            y_test_pred = model.predict(X_test)
            y_test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
            test_acc.append(y_test_accuracy)
            test_f1.append(metrics.f1_score(y_test, y_test_pred, average='weighted'))
            
            log_loss_curve = np.array(model.fitness_curve) * -1
            log_loss_curves.append(log_loss_curve)
            # print(model.fitted_weights.shape)
            # fitted_weights.append(model.fitted_weights)
            loss.append(model.loss)
            
        # print("train ", train_acc)
        # print("test", test_acc)
        # print(np.array(log_loss_curves).shape)
        return train_acc, train_f1, test_acc, test_f1, np.array(log_loss_curves), loss
