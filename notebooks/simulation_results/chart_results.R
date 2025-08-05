rm(list=ls())
# Load necessary libraries
# Libraries to import data
if(!require(readr)) install.packages('readr') 
if(!require(here)) install.packages('here') 
# Data manipulation
if(!require(dplyr)) install.packages('dplyr') 
if(!require(tidyverse)) install.packages('tidyverse')
if(!require(janitor)) install.packages('janitor')
if(!require(magrittr)) install.packages('magrittr') 
if(!require(stringr)) install.packages('stringr') 
# Data quality
if(!require(skimr)) install.packages('skimr') 
# Data visualisation
if(!require(ggplot2)) install.packages('ggplot2') 
if(!require(ggpubr)) install.packages('ggpubr') 
if(!require(gridExtra)) install.packages('gridExtra') 
if(!require(grid)) install.packages('grid') 
if(!require(hrbrthemes)) install.packages('hrbrthemes') 
if(!require(RColorBrewer)) install.packages('RColorBrewer') 
if(!require(scales)) install.packages('scales') 
if(!require(ggridges)) install.packages('ggridges') 
if(!require(reshape2)) install.packages('reshape2') 
if(!require(kableExtra)) install.packages('kableExtra') 
if(!require(formattable)) install.packages('formattable') 
# palette of colors
#Palette resource https://r-charts.com/color-palettes/#Google_vignette 
if(!require(wesanderson)) install.packages('wesanderson')
devtools::install_github("karthik/wesanderson")
# R-markdown style
if(!require(patchwork)) install.packages('patchwork') 
if(!require(kableExtra)) install.packages('kableExtra') 
if(!require(flextable)) install.packages('flextable') 
if(!require(knitr)) install.packages('knitr') 
# Genral color settings
light_blue = '#cbd9ed'
light_red = '#ffb089'
# My personalised palette
my_palette = c(wes_palette("Cavalcanti1")[3],wes_palette("FantasticFox1")[3],wes_palette("Cavalcanti1")[c(1,2,4)])
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
 p = ggplot(data, aes(x = Neighbors)) +
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
  
# Boxplots ----------------------------------------------------------------
if(!require(openxlsx)) install.packages('openxlsx')
if(!require(readxl)) install.packages('readxl')
df = read_excel("summary_results_boxplots.xlsx")
df$neighbor = factor(df$neighbor, levels = c('graph_neighbor_2', 'graph_neighbor_5', 
                                              'graph_neighbor_10', 'graph_neighbor_15', 
                                              'graph_neighbor_20'))

df$std =factor(df$std,
                      levels = df$std %>% unique() %>% sort(),
                      labels = gsub("_", ":", df$std %>% unique() %>% sort()))

baseline_df <- df %>%
  group_by(std, neighbor) %>%
  summarise(BE = mean(BE), .groups = 'drop')

# Box plot

# ----------------------------------------------------------------------------------
# ---------------------------std:1", "std:1.5-----------------------------------------
# ----------------------------------------------------------------------------------


p = ggplot(df %>% 
         filter( std %in% c( "std:1", "std:1.5")),
       aes(x = neighbor, y = SetRed)
       ) +
  geom_boxplot(fill = my_palette[1]) +
  # Red dashed baseline lines
  geom_hline(data = baseline_df %>% filter( std %in% c( "std:1", "std:1.5")),
             aes(yintercept = BE),
             color = "red", 
             linetype = "dashed", 
             linewidth = 0.8) +
  
  # Add baseline value text
  geom_text(data = baseline_df %>% filter( std %in% c( "std:1", "std:1.5")),
            aes(x = 1, y = BE,  # x = 1 means leftmost neighbor
                label = sprintf("Base Estimator Accuracy: %.2f", BE)),
            inherit.aes = FALSE,
            vjust = -1, hjust = 0,
            color = "red", size = 3.2, fontface = "italic") +
  
  facet_wrap(~std, ncol = 3) +#,scales = "free_y") +
  labs(title = "Boxplot of SETRED Accuracy: Standard Deviation 1 and 1.5",
       x = "Number of Neighbors",
       y = "Accuracy of SETRED") +
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "lightblue", color = "black"),
    panel.spacing = unit(1, "lines"),
    legend.position = 'bottom',
    legend.title.position = 'top',
    legend.title = element_text(size=12,hjust = .5, face = 'bold'),
    legend.margin=margin(0, 0, 0, 0),
    legend.text = element_text(size=12),
    legend.justification = "center",
    legend.direction = "horizontal",
    plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8)
  ) +
  scale_x_discrete(labels = c("graph_neighbor_2" = "2",
                              "graph_neighbor_5" = "5",
                              "graph_neighbor_10" = "10",
                              "graph_neighbor_15" = "15",
                              "graph_neighbor_20" = "20")) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(labels = label_number(accuracy = 0.01))

p

ggsave(filename = paste0("../../images/",  "boxplot_std15.png"), plot = p, width = 8, height = 6)

# ----------------------------------------------------------------------------------
# ---------------------------std:2", "std:3-----------------------------------------
# ----------------------------------------------------------------------------------

p =  ggplot(df %>% 
              filter( std %in% c( "std:2", "std:3")),
            aes(x = neighbor, y = SetRed)
) +
  geom_boxplot(fill = my_palette[1]) +
  # Red dashed baseline lines
  geom_hline(data = baseline_df %>% filter( std %in% c( "std:2", "std:3")),
             aes(yintercept = BE),
             color = "red", 
             linetype = "dashed", 
             linewidth = 0.8) +
  
  # Add baseline value text
  geom_text(data = baseline_df %>% filter( std %in% c( "std:2", "std:3")),
            aes(x = 1, y = BE,  # x = 1 means leftmost neighbor
                label = sprintf("Base Estimator Accuracy: %.2f", BE)),
            inherit.aes = FALSE,
            vjust = 1.8, hjust = 0,
            color = "red", size = 3.2, fontface = "italic") +
  
  facet_wrap(~std, ncol = 3) +#,scales = "free_y") +
  labs(title = "Boxplot of SETRED Accuracy: Standard Deviation 2 and 3",
       x = "Number of Neighbors",
       y = "Accuracy of SETRED") +
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "lightblue", color = "black"),
    panel.spacing = unit(1, "lines"),
    legend.position = 'bottom',
    legend.title.position = 'top',
    legend.title = element_text(size=12,hjust = .5, face = 'bold'),
    legend.margin=margin(0, 0, 0, 0),
    legend.text = element_text(size=12),
    legend.justification = "center",
    legend.direction = "horizontal",
    plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8)
  ) +
  scale_x_discrete(labels = c("graph_neighbor_2" = "2",
                              "graph_neighbor_5" = "5",
                              "graph_neighbor_10" = "10",
                              "graph_neighbor_15" = "15",
                              "graph_neighbor_20" = "20")) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(labels = label_number(accuracy = 0.01))
p
ggsave(filename = paste0("../../images/",  "boxplot_std3.png"), plot = p, width = 8, height = 6)

# ----------------------------------------------------------------------------------
# ---------------------------std:4", "std:5-----------------------------------------
# ----------------------------------------------------------------------------------

p =  ggplot(df %>% 
              filter( std %in% c( "std:4", "std:5")),
            aes(x = neighbor, y = SetRed)
) +
  geom_boxplot(fill = my_palette[1]) +
  # Red dashed baseline lines
  geom_hline(data = baseline_df %>% filter( std %in% c( "std:4", "std:5")),
             aes(yintercept = BE),
             color = "red", 
             linetype = "dashed", 
             linewidth = 0.8) +
  
  # Add baseline value text
  geom_text(data = baseline_df %>% filter( std %in% c( "std:4", "std:5")),
            aes(x = 1, y = BE,  # x = 1 means leftmost neighbor
                label = sprintf("Base Estimator Accuracy: %.2f", BE)),
            inherit.aes = FALSE,
            vjust = 2.0, hjust = 0.1,
            color = "red", size = 3.2, fontface = "italic") +
  
  facet_wrap(~std, ncol = 3) +#,scales = "free_y") +
  labs(title = "Boxplot of SETRED Accuracy: Standard Deviation 4 and 5",
       x = "Number of Neighbors",
       y = "Accuracy of SETRED") +
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "lightblue", color = "black"),
    panel.spacing = unit(1, "lines"),
    legend.position = 'bottom',
    legend.title.position = 'top',
    legend.title = element_text(size=12,hjust = .5, face = 'bold'),
    legend.margin=margin(0, 0, 0, 0),
    legend.text = element_text(size=12),
    legend.justification = "center",
    legend.direction = "horizontal",
    plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8)
  ) +
  scale_x_discrete(labels = c("graph_neighbor_2" = "2",
                              "graph_neighbor_5" = "5",
                              "graph_neighbor_10" = "10",
                              "graph_neighbor_15" = "15",
                              "graph_neighbor_20" = "20")) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(labels = label_number(accuracy = 0.01))

p
ggsave(filename = paste0("../../images/",  "boxplot_std5.png"), plot = p, width = 8, height = 6)



