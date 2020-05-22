import pandas as pd
import numpy as np
import random as rd
import matplotlib
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
#https://towardsdatascience.com/symbolic-regression-and-genetic-programming-8aed39e7f030
#https://towardsdatascience.com/understanding-random-forest-58381e0602d2


def gen_parametric_function_data(fun, timeline):
    return fun(timeline)


def noise(std, count):
    return np.random.normal(0, std, count)


def gen_validation_data(fun, start, end, points=1000, nstd=0.1, fault_count=50):
    random_start = rd.randint(start, end)
    tl = np.linspace(random_start, random_start + start, points)
    data = gen_parametric_function_data(fun, tl) + noise(nstd, points)
    labels = np.zeros(points)
    for i in range(fault_count):
        random_fault = rd.randint(0, points - 1)
        labels[random_fault] = 1.
        data[random_fault] *= np.random.uniform(-10, 10)
    return tl, data, labels


def calc_labels(limiar, recs, reals):
    calculated_labels = []
    for rec, real in zip(recs, reals):
        if abs(rec - real) > limiar:
            calculated_labels.append(1.)
        else:
            calculated_labels.append(0.)
    return calculated_labels


def calc_labels_fitness(limiar, metric='acc'):
    labels = calc_labels(limiar, reconstruction, validation_data)
    cm = confusion_matrix(real_labels, labels)
    if metric == 'acc':
        return -(cm[1][1] + cm[0][0]) / (cm[0][1] + cm[1][1] + cm[0][0] + cm[1][0])
    elif metric == 'recall':
        return -(cm[1][1]/(cm[0][1] + cm[1][1]))
    elif metric == 'precision':
        return -(cm[1][1]/(cm[1][0] + cm[1][1]))
    elif metric == 'f1':
        return -(2*((cm[1][1]/(cm[1][0] + cm[1][1]))*(cm[1][1]/(cm[0][1] + cm[1][1]))) /
                 ((cm[1][1]/(cm[1][0] + cm[1][1]))+(cm[1][1]/(cm[0][1] + cm[1][1]))))


def fun(x):
    # return np.sin(x) + np.cos(x) ** 2
    return 5*(x - np.floor(x))

# Params
parametric_function = fun
noise_std = 0.01
points_count = 5000
timeline_end = 25
validation_max_end = timeline_end + 1000
validation_fault_count = int(0.03 * points_count)
initial_fault_limiar = 0.2

# Timeline and original base function
timeline = np.linspace(0, timeline_end, points_count)
sample_data = gen_parametric_function_data(parametric_function, timeline) + noise(noise_std, points_count)
plt.plot(timeline, sample_data)
plt.show()

# Train/Test separation
X_train, X_test, y_train, y_test = train_test_split(timeline, sample_data, test_size=0.2)

# Apply regressor
reg = SymbolicRegressor(population_size=2000,
                        generations=20, stopping_criteria=0.01,
                        p_crossover=0.7, p_subtree_mutation=0.1,
                        p_hoist_mutation=0.05, p_point_mutation=0.1,
                        max_samples=0.9, verbose=1,
                        parsimony_coefficient=0.01, random_state=0,
                        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'tan', 'abs', 'log'))
reg.fit(X_train.reshape(-1, 1), y_train)
score = reg.score(X_test.reshape(-1, 1), y_test)
print("Function Regressed:", reg._program, " | Score:", score)

# Create validation data with fault and labels
validation_timeline, validation_data, real_labels = gen_validation_data(parametric_function,
                                                                        timeline_end,
                                                                        validation_max_end,
                                                                        points_count,
                                                                        noise_std,
                                                                        validation_fault_count)

# Validate fault detection
reconstruction = reg.predict(validation_timeline.reshape(-1, 1))
res = minimize(calc_labels_fitness, np.array([initial_fault_limiar]), method='Nelder-Mead')
rec_labels = calc_labels(res.x[0], reconstruction, validation_data)
print(confusion_matrix(real_labels, rec_labels))

# Plot results
fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(10)
# axs[0].plot(validation_timeline, validation_data)
axs[1].plot(validation_timeline, reconstruction)
fig.show()