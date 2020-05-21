import pandas as pd
import numpy as np
import random as rd
import matplotlib
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def noise_generation(amplitude, count):
    return np.random.normal(-amplitude, amplitude, count)


noise_amplitude = 0.05
end = 100
points_count = 1000

timeline = np.linspace(0, end, points_count)
sample_data = np.sin(timeline) + noise_generation(noise_amplitude, points_count)

plt.plot(timeline, sample_data, '.')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(timeline, sample_data, test_size=0.2)

reg = SymbolicRegressor(population_size=5000,
                        generations=20, stopping_criteria=0.01,
                        p_crossover=0.7, p_subtree_mutation=0.1,
                        p_hoist_mutation=0.05, p_point_mutation=0.1,
                        max_samples=0.9, verbose=1,
                        parsimony_coefficient=0.01, random_state=0)
reg.fit(X_train.reshape(-1, 1), y_train)
print(reg._program)

reg.predict(X_test.reshape(-1, 1))

est_tree = DecisionTreeRegressor()
est_tree.fit(X_train.reshape(-1, 1), y_train)
score_tree = est_tree.score(X_test.reshape(-1, 1), y_test)
est_rf = RandomForestRegressor()
est_rf.fit(X_train.reshape(-1, 1), y_train)
score_rf = est_rf.score(X_test.reshape(-1, 1), y_test)

est_rf.predict(0)

reconstruction = est_rf.predict(np.array(timeline).reshape(-1, 1))
plt.plot(timeline, reconstruction, '.')
plt.show()