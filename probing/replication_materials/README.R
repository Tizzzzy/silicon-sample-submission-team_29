
# README for replication materials for: 
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# Run the below code to reproduce the results in the main text and online appendix.
# Please note the WARNING about runtimes before executing the model fitting code.
# Please refer to the individual scripts for additional details.

# Load relevant libraries
library(tidyverse)
library(brms)
library(tidybayes)
library(estimatr)
library(cowplot)
library(broom)
library(cmdstanr)
library(MASS, exclude = "select")

# Set seed
set.seed(42)

# Read in the dta
df <- readRDS("data/data_RM.rds")

# 1. REPLICATION CODE FOR MODEL FITTING ----

# WARNING: 
# Most of the multilevel models were fitted on a high performance computing cluster with a minimum completion time of at least 1.5 hours.
# Thus, for convenience, the replication materials package already contains the raw model output (within the "cluster_output" sub-folders). 
# You can therefore skip Section 1 if you wish and proceed to the reproduction of the model analysis, figures, and tables (Section 2 below).

# 1.1. Primary multilevel model ----
#source("model_fitting/primary/model_fitting__primary.R")

# 1.2. Conditional average effects models ----
#source("model_fitting/moderators_conditional/model_fitting__moderators_conditional.R")

# 1.3. Interaction effects models ----
#source("model_fitting/moderators_interaction/model_fitting__moderators_interaction.R")

# 2. REPLICATION CODE FOR MODEL ANALYSIS, FIGURES, TABLES AND ROBUSTNESS CHECKS ----

# 2.1. Primary analysis ----
source("analysis_primary__RM.R")

# 2.2. Moderators analysis ----
source("analysis_moderators__RM.R")

# 2.3. Robustness analysis ----
source("analysis_robustness__RM.R")

# 2.4. ATE on the distribution analysis ----
source("analysis_ATE_on_distribution__RM.R")

# 2.5. Policy issue-level analysis ----
source("analysis_issue_level_fx__RM.R")

# 3. FINAL NOTES ----

# The dataset is "data_RM.rds".
# The codebook for the data is available in "codebook.xlsx".
