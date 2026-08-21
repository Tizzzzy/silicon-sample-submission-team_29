
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# MODERATORS ANALYSIS ####

# library(tidyverse)
# library(brms)
# library(tidybayes)
# library(estimatr)
# library(cowplot)

set.seed(42)

# Do you want to output model traceplots? (Takes a little while.)
output_traceplots <- FALSE

# Read in data ----
df <- readRDS("data/data_RM.rds")

# Wrangle data
df_for_model <-
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  drop_na(likertAgree_recoded)

# Conditional average effects models ----

# > Read in summary tables ----

# List file names
file_list_tables <- 
  list.files(path = "model_fitting/moderators_conditional/cluster_output/tables", pattern = "*.rds") %>% 
  as.list(.)

# Read in files
tables_conditional <- 
  map(file_list_tables,
      ~readRDS(paste0("model_fitting/moderators_conditional/cluster_output/tables/", .x)))

# Name and tidy
names(tables_conditional) <- file_list_tables 
names(tables_conditional) <- str_remove_all(names(tables_conditional), "summary_table__conditional_fx__")
names(tables_conditional) <- str_remove_all(names(tables_conditional), ".rds")

# > Read in samples ----

file_list_samples <- 
  list.files(path = "model_fitting/moderators_conditional/cluster_output/samples", pattern = "*.rds") %>% 
  as.list(.)

samples_conditional <- 
  map(file_list_samples,
      ~readRDS(paste0("model_fitting/moderators_conditional/cluster_output/samples/", .x)))

# Name and tidy
names(samples_conditional) <- file_list_samples
names(samples_conditional) <- str_remove_all(names(samples_conditional), "samples_pop__conditional_fx__")
names(samples_conditional) <- str_remove_all(names(samples_conditional), ".rds")

# > Summarize samples ----
summary_conditional <-
  imap(samples_conditional,
       function(.x, .y) {
         
         .x %>% 
           pivot_longer(cols = 3:ncol(.), names_to = "parameter") %>% 
           group_by(parameter) %>% 
           median_hdi(value, .width = 0.95) %>% 
           ungroup() %>% 
           mutate(model = .y)
         
       }) %>% 
  bind_rows()

# > Plot results ----

theme_set(
  theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text = element_text(color = "black")))

# Wrangle
summary_conditional <-
  summary_conditional %>% 
  mutate(parameter_label = case_when(parameter == "b_cue" ~ "ATE of\nParty Leader\nCue",
                                     parameter == "b_info" ~ "ATE of\nPersuasive\nMessage",
                                     parameter == "b_info:cue" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                     TRUE ~ parameter)) %>% 
  mutate(parameter_label = factor(parameter_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                              "ATE of\nPersuasive\nMessage",
                                                              "ATE of\nParty Leader\nCue"))) %>% 
  separate(model, c("moderator", "mod_level"), sep = "__") %>% 
  mutate(mod_level_label = case_when(moderator == "vote_party" ~ mod_level,
                                     moderator == "ba_degree" & mod_level == 0 ~ "No College Degree",
                                     moderator == "ba_degree" & mod_level == 1 ~ "College Degree",
                                     moderator == "strong_partisan" & mod_level == 0 ~ "Not Strong Partisan",
                                     moderator == "strong_partisan" & mod_level == 1 ~ "Strong Partisan",
                                     moderator == "female" & mod_level == 0 ~ "Not Female",
                                     moderator == "female" & mod_level == 1 ~ "Female",
                                     moderator == "cue_type" & mod_level == "one_sided" ~ "One-sided cue",
                                     moderator == "cue_type" & mod_level == "two_sided" ~ "Two-sided cue",
                                     str_detect(moderator, "tertiles") & mod_level == 1 ~ "Lowest Tertile",
                                     str_detect(moderator, "tertiles") & mod_level == 3 ~ "Highest Tertile"))

# Plot
plots_conditional <-
  map(unique(summary_conditional$moderator),
      function(.x) {
      
      if(.x == "vote_party") { level_colors <- c("blue", "red") }
      if(.x == "ba_degree") { level_colors <- c("purple", "magenta2") }
      if(.x == "strong_partisan") { level_colors <- c("darkgreen", "green3") }
      if(.x == "female") { level_colors <- c("black", "grey60") }
      if(.x == "PK_sum_tertiles") { level_colors <- c("turquoise3", "turquoise4") }
      if(.x == "age_tertiles") { level_colors <- c("dodgerblue", "dodgerblue4") }
      if(.x == "cue_type") { level_colors <- c("orange3", "black") }
      
        summary_conditional %>% 
        filter(parameter_label != "ATE of\nParty Leader\nCue") %>% 
        filter(moderator == .x) %>% 
        ggplot(aes(x = parameter_label, y = value, color = mod_level_label, shape = mod_level_label)) +
        geom_point(size = 2.5, position = position_dodge(.7)) +
        geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7, position = position_dodge(.7)) +
        coord_flip(ylim = c(-0.7, 0.25)) +
        labs(x = "", y = "Estimate (Median [95% HPDI])\n", 
             title = "Subgroup Conditional Average Effects" ) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
        geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4) +
        scale_color_manual(values = level_colors) +
        guides(color = guide_legend(reverse = T), shape = guide_legend(reverse = T)) +
        theme(legend.position = c(0.2, 0.25),
              legend.title = element_blank(),
              legend.box.background = element_rect(),
              plot.margin = unit(c(0,0,0,0), "cm"),
              legend.text = element_text(size = 10),
              axis.text = element_text(size = 12),
              axis.title.x = element_text(size = 12))
      
    })

names(plots_conditional) <- unique(summary_conditional$moderator)

# Interaction models ----

list_moderators <- c("trump_republican_c",
                     "ba_degree_c",
                     "PK_sum_z",
                     "strong_partisan_c",
                     "age_z",
                     "female_c",
                     "two_sided_cue_c")

# > Read in samples ----
samples_interaction <-
  map(list_moderators,
      function(.x) { readRDS(paste0("model_fitting/moderators_interaction/cluster_output/samples/samples_pop__interaction_model__", .x, ".rds")) })

names(samples_interaction) <- list_moderators # label samples to keep track

# > Read in summary tables ----
tables_interaction <-
  map(list_moderators,
      function(.x) { readRDS(paste0("model_fitting/moderators_interaction/cluster_output/tables/summary_table__interaction_model__", .x, ".rds")) })

names(tables_interaction) <- list_moderators 

# > Summarize samples ----
summary_interactions <-
  map(list_moderators,
      function(.x) {
        
        samples_interaction[[.x]] %>% 
          pivot_longer(cols = 3:ncol(.),
                       names_to = "parameter") %>% 
          group_by(parameter) %>% 
          median_hdi(value, .width = 0.95) %>% 
          ungroup() %>% 
          mutate(moderator = .x)
        
      }) %>% 
  bind_rows()

# Wrangle
summary_interactions <-
  summary_interactions %>% 
  mutate(difference_test_label = case_when(parameter == "b_cue:mod" ~ "ATE of\nParty Leader\nCue",
                                           parameter == "b_info:mod" ~ "ATE of\nPersuasive\nMessage",
                                           parameter == "b_info:cue:mod" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                           TRUE ~ parameter)) %>% 
  mutate(difference_test_label = factor(difference_test_label, 
                                        levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                   "ATE of\nPersuasive\nMessage",
                                                   "ATE of\nParty Leader\nCue")),
         moderator_label = case_when(moderator == "trump_republican_c" ~ "Partisan Identity",
                                     moderator == "ba_degree_c" ~ "Educational Attainment",
                                     moderator == "PK_sum_z" ~ "Political Knowledge",
                                     moderator == "strong_partisan_c" ~ "Strength of Partisanship",
                                     moderator == "age_z" ~ "Age",
                                     moderator == "female_c" ~ "Gender",
                                     moderator == "two_sided_cue_c" ~ "Cue Environment"))

plots_interactions <-
  map(unique(summary_interactions$moderator),
      function(.x) {
        
        summary_interactions %>% 
          filter(difference_test_label != "ATE of\nParty Leader\nCue") %>% 
          drop_na(difference_test_label) %>% 
          filter(moderator == .x) %>% 
          ggplot(aes(x = difference_test_label, y = value)) +
          geom_point(size = 2.5, position = position_dodge(.7), shape = "square") +
          geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7, position = position_dodge(.7)) +
          coord_flip(ylim = c(-0.4, 0.3)) +
          labs(x = "", y = "Estimate (Median [95% HPDI])\n", 
               title = "Interaction (Difference) Estimates") +
          geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
          facet_wrap(~moderator_label, strip.position = "right") +
          geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4) +
          theme(axis.text.y = element_blank(),
                axis.ticks.y = element_blank(),
                strip.text = element_text(size = 12, face = "bold"),
                strip.background = element_blank(),
                plot.margin = unit(c(0,0,0,0), "cm"),
                axis.title.x = element_text(size = 12), 
                axis.text = element_text(size = 12))
        
      })

names(plots_interactions) <- list_moderators

# Join together all plots ----
g <-
  plot_grid(
  
  # Party ID
  plots_conditional$vote_party          + labs(y = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  plots_interactions$trump_republican_c + labs(y = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
       
  # Party strength   
  plots_conditional$strong_partisan    + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  plots_interactions$strong_partisan_c + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
          
  # Education   
  plots_conditional$ba_degree    + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  plots_interactions$ba_degree_c + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  
  # Political knowledge   
  plots_conditional$PK_sum_tertiles + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  plots_interactions$PK_sum_z       + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  
  # Age   
  plots_conditional$age_tertiles + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  plots_interactions$age_z       + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), 
  
  # Gender   
  plots_conditional$female    + labs(title = "") + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()),
  plots_interactions$female_c + labs(title = "") + labs(y = "", title = "") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()),
  
  # Cue environment   
  plots_conditional$cue_type         + labs(title = ""),
  plots_interactions$two_sided_cue_c + labs(title = ""),
          
  ncol = 2)

ggsave(plot = g, filename = "figures/figure_5.png", height = 17, width = 12, dpi = 300)
ggsave(plot = g, filename = "figures/figure_5.pdf", height = 17, width = 12, dpi = 300)


# Output model traceplots ----

# To save time, only output traceplots if TRUE
if(output_traceplots == TRUE) {

# > Traceplots ----

# Write function to make traceplots for parameters we care about
fun_traceplots <- function(param, samples = NULL) {
  
  samples %>%
    rename(value = !!as.name(param)) %>%
    ggplot(aes(x = iter - 1000, y = value, color = as.factor(chain))) + 
    theme_bw() +
    geom_line(alpha = 0.8) +
    labs(title = param, x = "iteration") +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, face = "bold", size = 8),
          axis.text = element_text(color = "black"))
  
}

# >> Conditional average effects models ----

traceplots_conditional <-
  map(samples_conditional,
      function(.y) {
        
        samples <- .y
        
        # List parameters
        params <- 
          samples %>% 
          select(-c("chain", "iter")) %>% 
          names()
        
        # Iterate over parameters making traceplots
        map(params,
            ~fun_traceplots(param = .x, samples = samples))
        
      })

names(traceplots_conditional) <- names(samples_conditional)

# Save plots
imap(traceplots_conditional,
     function(.x, .y) {
       
       ggsave(paste0("appendix/figures/traceplots__moderators__conditional__", .y, ".png"),
                     plot = plot_grid(plotlist = .x),
                     dpi = 300, height = 10, width = 10)
       
     })

# >> Interaction models ----

traceplots_interaction <-
  map(samples_interaction,
      function(.y) {
        
        samples <- .y
        
        # List parameters
        params <- 
          samples %>% 
          select(-c("chain", "iter")) %>% 
          names()
        
        # Iterate over parameters making traceplots
        map(params,
            ~fun_traceplots(param = .x, samples = samples))
        
      })

names(traceplots_interaction) <- names(samples_interaction)

# Save plots
imap(traceplots_interaction,
     function(.x, .y) {
       
       ggsave(paste0("appendix/figures/traceplots__moderators__interaction__", .y, ".png"),
              plot = plot_grid(plotlist = .x),
              dpi = 300, height = 16, width = 20)
       
     })

}

# > Tables ----

# Conditional average effects models
imap(tables_conditional,
     function(.x, .y) {
      
       tab_out <-
         .x %>% 
         select(Group:Rhat) %>% 
         mutate(Group = case_when(Group == "subject_id" ~ "Respondent",
                                  Group == "item_label" ~ "Policy question",
                                  TRUE ~ str_to_sentence(Group)))
       
       colnames(tab_out) <- c("Group", "Term", "Estimate", "Est. Error",
                              "L. 95\\% CI", "H. 95\\% CI", "Eff. Samples", "$\\hat{R}$")
       
       saveRDS(tab_out, paste0("appendix/tables/table__moderators__conditional__", .y, ".rds"))
      
     })

# Interaction models
imap(tables_interaction,
     function(.x, .y) {
       
       tab_out <- 
         .x %>% 
         select(Group:Rhat) %>% 
         mutate(Group = case_when(Group == "subject_id" ~ "Respondent",
                                  Group == "item_label" ~ "Policy question",
                                  TRUE ~ str_to_sentence(Group)))
       
       colnames(tab_out) <- c("Group", "Term", "Estimate", "Est. Error",
                              "L. 95\\% CI", "H. 95\\% CI", "Eff. Samples", "$\\hat{R}$")
       
       saveRDS(tab_out, paste0("appendix/tables/table__moderators__interaction__", .y, ".rds"))
       
     })


# Plots with party cue ATE included ----

plots_conditional__with_cue <-
  map(unique(summary_conditional$moderator),
      function(.x) {
        
        if(.x == "vote_party") { level_colors <- c("blue", "red") }
        if(.x == "ba_degree") { level_colors <- c("purple", "magenta2") }
        if(.x == "strong_partisan") { level_colors <- c("darkgreen", "green3") }
        if(.x == "female") { level_colors <- c("black", "grey60") }
        if(.x == "PK_sum_tertiles") { level_colors <- c("turquoise3", "turquoise4") }
        if(.x == "age_tertiles") { level_colors <- c("dodgerblue", "dodgerblue4") }
        if(.x == "cue_type") { level_colors <- c("orange3", "black") }
        
        summary_conditional %>% 
          filter(!is.na(parameter_label)) %>% 
          filter(moderator == .x) %>% 
          ggplot(aes(x = parameter_label, y = value, color = mod_level_label, shape = mod_level_label)) +
          geom_point(size = 2.5, position = position_dodge(.7)) +
          geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7, position = position_dodge(.7)) +
          coord_flip(ylim = c(-0.7, 0.7)) +
          labs(x = "", y = "Estimate (Median [95% HPDI])\n", 
               title = "Subgroup Conditional Average Effects" ) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
          geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4) +
          scale_color_manual(values = level_colors) +
          guides(color = guide_legend(reverse = T), shape = guide_legend(reverse = T)) +
          theme(legend.position = c(0.2, 0.9),
                legend.title = element_blank(),
                legend.box.background = element_rect(),
                plot.margin = unit(c(0,0,0,0), "cm"),
                legend.text = element_text(size = 10),
                axis.text = element_text(size = 12),
                axis.title.x = element_text(size = 12))
        
      })

names(plots_conditional__with_cue) <- unique(summary_conditional$moderator)

plots_interactions__with_cue <-
  map(unique(summary_interactions$moderator),
      function(.x) {
        
        summary_interactions %>% 
          filter(!is.na(difference_test_label)) %>% 
          drop_na(difference_test_label) %>% 
          filter(moderator == .x) %>% 
          ggplot(aes(x = difference_test_label, y = value)) +
          geom_point(size = 2.5, position = position_dodge(.7), shape = "square") +
          geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7, position = position_dodge(.7)) +
          coord_flip(ylim = c(-0.4, 0.4)) +
          labs(x = "", y = "Estimate (Median [95% HPDI])\n", 
               title = "Interaction (Difference) Estimates") +
          geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
          facet_wrap(~moderator_label, strip.position = "right") +
          geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4) +
          theme(axis.text.y = element_blank(),
                axis.ticks.y = element_blank(),
                strip.text = element_text(size = 12, face = "bold"),
                strip.background = element_blank(),
                plot.margin = unit(c(0,0,0,0), "cm"),
                axis.title.x = element_text(size = 12), 
                axis.text = element_text(size = 12))
        
      })

# Name and reorder to match conditional plots
names(plots_interactions__with_cue) <- list_moderators
plots_interactions__with_cue <- plots_interactions__with_cue[c("age_z", "ba_degree_c", "two_sided_cue_c", "female_c", "PK_sum_z", "strong_partisan_c", "trump_republican_c")]

# Join plots and save
pmap(list(plots_conditional__with_cue, plots_interactions__with_cue, names(plots_interactions__with_cue)),
     function(...) {
       
       ggsave(paste0("appendix/figures/results__moderators_with_cue__", ..3, ".png"), 
              plot = plot_grid(..1, ..2, ncol = 2),
              dpi = 300, height = 6, width = 12)
       
     })


# Table for main text ----

# > Join median/HPDI summaries with diagnostics ----

# First for the conditional models
con_x <- 
  summary_conditional %>% 
  filter(str_detect(parameter, "b_"), 
         parameter != "b_Intercept") %>% 
  arrange(moderator, parameter) %>% 
  select(moderator, mod_level_label, mod_level, parameter, parameter_label, value, .lower, .upper)

con_y <-
  map(tables_conditional,
      function(.x) { .x %>% mutate(moderator_level = as.character(moderator_level)) }) %>% 
  bind_rows() %>% 
  filter(Term %in% c("info", "cue", "info:cue")) %>% 
  mutate(Term = str_c("b_", Term)) %>% 
  rename(mod_level = moderator_level,
         parameter = Term)

con_z <- 
  left_join(con_x,
            con_y %>% select(moderator, mod_level, parameter, Eff.Sample, Rhat),
            by = c("moderator", "mod_level", "parameter")) %>% 
  mutate(model = "Subgroup") %>% 
  select(moderator, parameter, value, .lower, .upper, Eff.Sample, Rhat, model, everything())

# Then for the interaction models
int_x <- 
  summary_interactions %>% 
  filter(parameter %in% c("b_info:mod", "b_cue:mod", "b_info:cue:mod")) %>% 
  arrange(moderator, parameter) %>% 
  select(moderator, moderator_label, parameter, value, .lower, .upper)

int_y <-
  tables_interaction %>% 
  bind_rows() %>% 
  filter(Term %in% c("info:mod", "cue:mod", "info:cue:mod")) %>% 
  mutate(Term = str_c("b_", Term)) %>% 
  rename(parameter = Term)

int_z <-
  left_join(int_x,
            int_y %>% select(moderator, parameter, Eff.Sample, Rhat),
            by = c("moderator", "parameter")) %>% 
  mutate(model = "Difference",
         parameter = str_remove_all(parameter, ":mod")) %>% 
  select(moderator, parameter, value, .lower, .upper, Eff.Sample, Rhat, model)

# > Join and wrangle ----
out <-
  bind_rows(con_z, int_z) %>% 
  arrange(moderator, parameter, model)

mod_names <-
  c("age_tertiles" = "Age", "age_z" = "Age",
    "ba_degree" = "Educational Attainment", "ba_degree_c" = "Educational Attainment",
    "cue_type" = "Cue Environment", "two_sided_cue_c" = "Cue Environment",
    "female" = "Gender", "female_c" = "Gender",
    "PK_sum_tertiles" = "Political Knowledge", "PK_sum_z" = "Political Knowledge",
    "strong_partisan" = "Strength of Partisanship", "strong_partisan_c" = "Strength of Partisanship",
    "trump_republican_c" = "Partisan Identity", "vote_party" = "Partisan Identity")

out <-
  out %>% 
  mutate(moderator = str_replace_all(moderator, mod_names)) %>% 
  mutate(moderator = str_remove_all(moderator, "_c")) %>% 
  arrange(moderator, parameter, fct_rev(model)) %>% 
  filter(parameter != "b_cue") %>% 
  mutate(parameter = case_when(parameter == "b_info" ~ "Message ATE",
                               parameter == "b_info:cue" ~ "Change in ATE under Party Cue"),
         estimate = paste0(format(round(value, 2), nsmall = 2), " [", 
                           format(round(.lower, 2), nsmall = 2), ", ", 
                           format(round(.upper, 2), nsmall = 2), "]")) %>% 
  mutate(value = format(round(value, 2), nsmall = 2),
         .lower = format(round(.lower, 2), nsmall = 2),
         .upper = format(round(.upper, 2), nsmall = 2),
         Eff.Sample = round(Eff.Sample, 0),
         Rhat = format(round(Rhat, 2), nsmall = 2)) %>% 
  select(moderator, parameter, model, mod_level_label, value, .lower, .upper)

names(out) <- c("Covariate", "Parameter", "Model", "Subgroup value", "Estimate", "Lower 95% HPDI", "Upper 95% HPDI")

# Write to file
write.csv(out, "tables/table__moderators__main_text.csv", row.names = F)
