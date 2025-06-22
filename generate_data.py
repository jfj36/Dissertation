from setred_package import simulated_data
import os
print(f"Running in {os.getcwd()}")
# Parameters of the simulation
n = 13000  # Number of samples
K = 2     # Number of classes
p = 2      # Number of features
kcenters = 8
# Generate data
X_ori, y_ori, X, y, X_unlabel, y_unlabel = simulated_data.create_data(
    n=n, kcenters=kcenters, K=K, p=p, std=2,
    export_path="data/"
)