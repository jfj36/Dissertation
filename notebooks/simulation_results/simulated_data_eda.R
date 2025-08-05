rm(list=ls())

# Library for plotting
library(ggplot2) 
library(gridExtra)
library(ggridges)
library(GGally)
library(kableExtra) 
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

setwd("~/Library/CloudStorage/OneDrive-UniversityofBath/00_Bath_Master/03_Dissertation/src/Goal1/setred/notebooks/simulation_results")

lfiles = list.files('../../data')
std_folder = lfiles[grep("std", lfiles)]
full_data = NULL
# I need to bind the data from all the standard folders adn sotre it in a full_data data frame
for (std in std_folder) {
  # Read the data
  data_ori = read.csv(paste0("../../data/", std,"/df_ori.csv"))
  # Convert target to factor for classification
  data_ori = data_ori %>% 
              mutate(target = as.factor(target))
  # Add a column for the standard folder name
  data_ori$std = std
  # Bind the data to the full_data data frame
  if (!exists("full_data")) {
    full_data = data_ori
  } else {
    full_data = rbind(full_data, data_ori)
  }
}


full_data$std =factor(full_data$std,
       levels = full_data$std %>% unique() %>% sort(),
       labels = gsub("_", ":", full_data$std %>% unique() %>% sort()))



# I need to diplay the scatter plot of the first two variables of the data in each standard folder
p = ggplot(full_data, aes(x = X0, y = X1, color = target)) +
  geom_point(alpha = 0.5) +
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
    plot.title = element_text( size = 12, face = "bold", hjust = 0.5),
    axis.title = element_text( size = 10, face = "bold"),
    axis.text.x = element_text(size = 8),
    axis.text.y  = element_text(size = 8)
  ) +
  facet_wrap(~std, ncol = 2) +
  scale_color_manual(values = my_palette, name = "Target Class") +
  guides(color = guide_legend(override.aes = list(shape = 15, size = 5))) +
  labs(x="X1",
       y = "X2",
       title="Scatterplot of Simulated Data by Standard Deviation"
       #subtitle="First two features (X0 and X1) across different standard deviations",
      
  )
p

ggsave(filename = paste0("../../images/",  "all_scatterplot.png"), plot = p, width = 8, height = 6)


 # GGpairs code


for (std in std_folder) {
  # Read the data
  data_ori = read.csv(paste0("../../data/", std,"/df_ori.csv"))
  data_ori = data_ori %>% 
              mutate(target = as.factor(target)) 
  # Create a scatter plot
  p = data_ori %>%
    ggpairs(columns = 1:2,
            ggplot2::aes(color = target, alpha=0.5),
            legend = 1,
            title = paste("Scatterplot of Simulated Data -", gsub('_',':', std) ),        
            upper = list(continuous = "points")) +
    theme_bw() +
    theme(
      strip.background = element_rect(fill = "lightblue", color = "black"),
      strip.text = element_text(color = "black", face = "bold"),
      #plot.margin = margin(-30,10,0,-10),
      panel.spacing = unit(1,'lines'),
      legend.position = 'none',
      legend.title.align = 0,
      legend.title.position = 'top',
      legend.title = element_blank(),
      legend.margin=margin(0, 0, 0, 0),
      legend.text = element_text(size=8),
      legend.justification = "center",
      legend.direction = "horizontal",
      plot.title = element_text(hjust = 0.5,size = 12, face = "bold"),
      axis.title = element_text( size = 12, face = "bold"),
      axis.text.x = element_text(size = 8)) +
    scale_color_manual(values = my_palette) +
    scale_fill_manual(values = my_palette)
  
  # Save the plot as an image
  ggsave(filename = paste0("../../images/", std, "_scatterplot.png"), plot = p, width = 8, height = 6)
  
}
 
# Remove the underscore from the folder name

# Scatterplots ------------------------------------------------------------
data_ori = read.csv("../../data/std_1/df_ori.csv")
data_ori = data_ori %>% 
  mutate(target = as.factor(target)) # Convert target to factor for classification
# Semi Supervised data
data_ss = read.csv("../../data/df_X.csv")
data_ss = data_ss %>% 
  mutate(target = as.factor(target)) # Convert target to factor for classification
# Unlabel data
data_label = read.csv("../../data/df_unlabel.csv")
data_label = data_label %>% 
  mutate(target = as.factor(target)) # Convert target to factor for classification
# Test data
data_test = read.csv("../../data/df_test.csv")
data_test = data_test %>% 
  mutate(target = as.factor(target))


# Scatter plots -----------------------------------------------------------
 p = data_ori %>%
      ggpairs(columns = 1:2,
              aes(color = target),
              legend = 1,
              title = "Scatterplot of Simulated Data",        
              upper = list(continuous = "points")
              ) +
      theme_bw() +
  theme(
    strip.background = element_rect(fill = "lightblue", color = "black"),
    strip.text = element_text(color = "black", face = "bold"),
    #plot.margin = margin(-30,10,0,-10),
    panel.spacing = unit(1,'lines'),
    legend.position = 'bottom',
    legend.title.align = 0,
    legend.title.position = 'top',
    legend.title = element_blank(),
    legend.margin=margin(0, 0, 0, 0),
    legend.text = element_text(size=8),
    legend.justification = "center",
    legend.direction = "horizontal",
    plot.title = element_text(hjust = 0.5),
    axis.title = element_text( size = 12, face = "bold"),
    axis.text.x = element_text(size = 8)) +
    scale_color_manual(values = my_palette) +
    scale_fill_manual(values = my_palette) 
p






