import mlrose 
import numpy as np
from sklearn import metrics 
from helpers import algos, data_helper, plot_helper, model_helper

def ga(X, y, lr, title, filename, max_iters=1000, pop_size=200, mutation_prob=0.1, runs=10):
    nn = algos.NN(
        X, y,
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'genetic_alg', 
        max_iters = max_iters,
        learning_rate = lr,
        clip_max = 50, 
        pop_size = pop_size,
        mutation_prob = mutation_prob
    )
    train_acc, train_f1, test_acc, test_f1, log_loss_curves, loss = nn.run(runs)
    # save curves
    fn = "results/{}.csv".format(filename)
    np.savetxt(fn, log_loss_curves, delimiter=",")
    plot_helper.plot_curves(log_loss_curves, title, x_label="Iterations", y_label="log loss", filename=filename)
    notes = """
        1 layer, 4 hidden nodes  
        clip max  = 50 
        activation = relu 
        max iters = {max_iters}
        learning rate = {lr}
        pop_size = {pop_size}
        mutation_prob = {m_prob}
    """.format(max_iters=max_iters, lr=lr, pop_size=pop_size, m_prob=mutation_prob)
    model_helper.save_nn_report(title, filename, train_acc, train_f1, test_acc, test_f1, loss, notes)


def ga_tuning_low_pop(X, y, max_iters=2000):
    lr = 0.0001
    # remember weights len = 196 so this is like 
    # wl, wl*5, wl*10, wl*15
    pop_size = [20, 200]
    mutation_probs = [0.1, 0.01, 0.001]
    for p in pop_size: 
        for m in mutation_probs:
            lr_str = str(lr).split(".")[1]
            m_str = str(m).split(".")[1]
            ga(X, y, 
                lr=lr, max_iters=max_iters, 
                pop_size =  p,
                mutation_prob = m,
                title="NN GA m_prob=0.{m} pop_size={p}".format(lr=lr_str, m=m_str, p=p), 
                filename="nn/ga/m{m}p{p}".format(lr=lr_str, m=m_str, p=p)
                )    
                
def ga_tuning_high_pop(X, y, max_iters=2000):
    lr = 0.0001
    # remember weights len = 196 so this is like 
    # wl, wl*5, wl*10, wl*15
    pop_size = [2000, 3000]
    mutation_probs = [0.1, 0.01, 0.001]
    for p in pop_size: 
        for m in mutation_probs:
            lr_str = str(lr).split(".")[1]
            m_str = str(m).split(".")[1]
            ga(X, y, 
                lr=lr, max_iters=max_iters, 
                pop_size =  p,
                mutation_prob = m,
                title="NN GA m_prob=0.{m} pop_size={p}".format(lr=lr_str, m=m_str, p=p), 
                filename="nn/ga/m{m}p{p}".format(lr=lr_str, m=m_str, p=p)
                )      
                    
if __name__ == "__main__":
    X, x_names, y = data_helper.get_mushroom_data("odor")
    
    # top left started 4:16
    # ga_tuning_low_pop(X, y, max_iters=1000)
    # bottom left started 4:17
    ga_tuning_high_pop(X, y, max_iters=2000)
    
    # trying a really small pop size (with 196 weights, 20 = approx 0.1*W)
    # the resulting accuracy scores are low as expected,  but better than chance!
    # top left -- took approx 20min , started 6:20 ~ about 20min
    # ga(X, y, lr=0.01, 
    #     title="NN GA lr=0.01 m_prob=0.1 pop_size=20", filename="nn/ga/vfastplz-lr01m1p20", 
    #     max_iters=1000, 
    #     pop_size=20, 
    #     mutation_prob=0.1)
    
    # # try with lower lr to rule out 
    # bottom left  -- started 6:20 ~ about 20min
    # ga(X, y, lr=0.001, 
    #     title="NN GA lr=0.001 m_prob=0.1 pop_size=20", filename="nn/ga/vfastplz-lr001m1p20", 
    #     max_iters=1000, 
    #     pop_size=20, 
    #     mutation_prob=0.1)

    # # try with higher lr to rule out 
    # decrease iters to 500 because should learn faster
    # top right  -- started 6:30
    # ga(X, y, lr=0.1, 
    #     title="NN GA lr=0.1 m_prob=0.1 pop_size=20", filename="nn/ga/vfastplz-lr1m1p20i500", 
    #     max_iters=500, 
    #     pop_size=20, 
    #     mutation_prob=0.1)
    
    # # try with lower lr because .001 looked less flat 
    # top left  -- started 6:44 
    # ga(X, y, lr=0.0001, 
    #     title="NN GA lr=0.0001 m_prob=0.1 pop_size=20", filename="nn/ga/vfastplz-lr0001m1p20i2k", 
    #     max_iters=2000, 
    #     pop_size=20, 
    #     mutation_prob=0.1)

    # lr = .1 = faster learning 
    # increase pop size to 200 (same as # of weights)
    # increase iterations too because we know increase pop => increase iters
    # bottom right -- started 6:33
    # ga(X, y, lr=0.1, 
    #     title="NN GA lr=0.1 m_prob=0.1 pop_size=200", filename="nn/ga/lr1m1p200i2k", 
    #     max_iters=2000, 
    #     pop_size=200, 
    #     mutation_prob=0.1)
    
    # try lower lr  
    # increase pop size to 200 (same as # of weights)
    # increase iterations too because we know increase pop => increase iters
    # bottom left -- started 6:47
    # ga(X, y, lr=0.001, 
    #     title="NN GA lr=0.001 m_prob=0.1 pop_size=200", filename="nn/ga/lr001m1p200i2k", 
    #     max_iters=2000, 
    #     pop_size=200, 
    #     mutation_prob=0.1)
        
    # try lower mutation 
    # pop size to 200 (same as # of weights)
    # less iterations for speed
    # top  right -- started 6:57 ~ 2 hrs 
    # ga(X, y, lr=0.001, 
    #     title="NN GA lr=0.001 m_prob=0.01 pop_size=200", filename="nn/ga/lr001m01p200i500", 
    #     max_iters=500, 
    #     pop_size=200, 
    #     mutation_prob=0.01)
    
    # try lower mutation 
    # pop size to 200 (same as # of weights)
    # less runs for speed
    # top  right -- started 9:04
    # ga(X, y, lr=0.001, 
    #     title="NN GA lr=0.001 m_prob=0.01 pop_size=200", filename="nn/ga/lr001m01p200i2kruns3", 
    #     max_iters=2000, 
    #     pop_size=200, 
    #     mutation_prob=0.01,
    #     runs=3)
    
    # pop size to 2000 
    # less runs for speed
    # top  left -- started 9:06
    # ga(X, y, lr=0.001, 
    #     title="NN GA lr=0.001 m_prob=0.1 pop_size=2000", filename="nn/ga/lr001m01p2ki2kruns3", 
    #     max_iters=2000, 
    #     pop_size=2000, 
    #     mutation_prob=0.1,
    #     runs=3)
        
    
    


    