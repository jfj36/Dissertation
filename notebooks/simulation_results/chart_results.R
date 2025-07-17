rm(list=ls())
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
# Load the dataset
setwd("~/Library/CloudStorage/OneDrive-UniversityofBath/00_Bath_Master/03_Dissertation/src/Goal1/setred/notebooks/simulation_results")
list.files()
data <- read.csv("chart_results.csv")
colnames(data) = c("Std", "Neighbors", "BE", "Diff", "Diff_percentage", "SETRED")
# order the levels of th Neighbors factor 2, 3, 5, 7, 9, 11
data$Neighbors <- factor(data$Neighbors, levels = c('neighbor_2', 'neighbor_5', 'neighbor_10', 'neighbor_15', 'neighbor_20'))


# Plot the data Neighbors vs Diff_percentage for each standard deviation

ggplot(data, aes(x = Neighbors, y = Diff_percentage)) +
  geom_line() +
  geom_point() +
  labs(title = "",
       x = "Number of Neighbors",
       y = "Performance improvements of Setred") +
  theme_bw() +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(-0.5, 15),
                     breaks = seq(-0.5, 15, by = 2),
                     labels = scales::percent_format(scale = 1)) + 
  scale_x_discrete(labels = c("neighbor_2" = "2",
                               "neighbor_5" = "5",
                               "neighbor_10" = "10",
                               "neighbor_15" = "15",
                               "neighbor_20" = "20")) +
  facet_wrap(~ Std) +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "Standard Deviation")) +
  theme(plot.title = element_text(hjust = 0.5))
  
# Plot Be and Setred vs neigbors for each standard deviation
ggplot(data, aes(x = Neighbors)) +
  geom_line(aes(y = BE, color = "Base Classifier")) +
  geom_line(aes(y = SETRED, color = "SETRED")) +
  geom_point(aes(y = BE, color = "Base Classifier")) +
  geom_point(aes(y = SETRED, color = "SETRED")) +
  labs(title = "Accuracy Metric",
       x = "Number of Neighbors",
       y = "Value") +
  theme_bw() +
  scale_color_brewer(palette = "Set1") +
  scale_x_discrete(labels = c("neighbor_2" = "2",
                               "neighbor_5" = "5",
                               "neighbor_10" = "10",
                               "neighbor_15" = "15",
                               "neighbor_20" = "20")) +
  facet_wrap(~ Std) +
  # Locate the title in the center
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "")) +
  theme(plot.title = element_text(hjust = 0.5))
  

