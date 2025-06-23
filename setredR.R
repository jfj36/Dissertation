rm(list=ls())

# Load libraries
library(ggplot2)
library(dplyr)
library(ssc)
library(tidyverse)
library(caret)

# Load data
# Set working directory
#setwd("~/Library/CloudStorage/OneDrive-UniversityofBath/00_Bath_Master/03_Dissertation/src/Goal1/setred")

# List files in the data directory
list.files('data')
# Load test data
test_data = read.csv("data/df_test.csv")
# Load SSL data
ssl_data = read.csv("data/df_X.csv")
ssl_data = ssl_data %>% mutate(target = ifelse(target == -1,NA, target))

# Target
cls <- which(colnames(ssl_data) == "target")

# Data preparation
## train
xtrain =  ssl_data[, -cls] 
ytrain = ssl_data[, cls]
## test
xitest = test_data[, -cls]
yitest = test_data[, cls]

# Model
m1 <- setred(x = xtrain, y = ytrain, dist = "euclidean", 
             learner = caret::knn3, 
             learner.pars = list(k = 1),
             pred = "predict")

# Prediction
pred1 <- predict(m1, as.matrix(xitest))
table(pred1, yitest)
confusionMatrix(data = pred1, reference = as.factor(yitest))

m1$model



getAnywhere(setredG)
getAnywhere(normalCriterion)





