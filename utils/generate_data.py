import os,sys
sys.path.append(os.path.abspath(".."))
from setred_package import simulated_data

print(f"Running in {os.getcwd()}")
# Parameters of the simulation
n = 13000  # Number of samples
K = 4     # Number of classes
p = 2      # Number of features
kcenters = 8
# Generate data
X_ori, y_ori, X, y, X_unlabel, y_unlabel, _,_ = simulated_data.create_data(
    n=n, kcenters=kcenters, K=K, p=p, std=1.5,
    export_path="../data/"
)