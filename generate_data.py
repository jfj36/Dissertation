from setred_package import simulated_data

# Parameters of the simulation
n = 13000 # Number of samples
K = 5 # Number of classes 5 are the number of types of myositis
p = 2 # Number of features


X_ori, y_ori, X, y, X_unlabel, y_unlabel = simulated_data.create_data(
    n=n, kcenters=5, K=K, p=p, std= 1,
    export_path="data/",
)