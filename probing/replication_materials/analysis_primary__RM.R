
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# PRIMARY ANALYSIS ####

# library(tidyverse)
# library(brms)
# library(tidybayes)
# library(estimatr)
# library(cowplot)
# library(broom)
# library(cmdstanr)

set.seed(42)

# Read in data and samples ----

df <- readRDS("data/data_RM.rds")
df_for_model <- df %>% filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% drop_na(likertAgree_recoded)

samples_pop__primary  <- readRDS("model_fitting/primary/cluster_output/samples/samples_pop__primary.rds")

# Summarize samples ----

# > Population samples ----
summary_pop__primary <-
  samples_pop__primary %>% 
  mutate(b_info_under_cue = b_info + `b_info:cue`) %>% 
  pivot_longer(cols = 3:ncol(.),
               names_to = "parameter") %>% 
  group_by(parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup()

# Plots ----

theme_set(
  theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text = element_text(color = "black"))
)

# > Raw data ----

# Compute raw condition means
critval <- qnorm(.025, lower.tail = F)
condition_means <-
  df_for_model %>% 
  group_by(condition) %>% 
  summarise(m_y = mean(likertAgree_recoded, na.rm = TRUE),
            n = n(),
            sd_y = sd(likertAgree_recoded, na.rm = TRUE)) %>% 
  ungroup() %>% 
  mutate(se = sd_y / (sqrt(n))) %>% 
  mutate(lwr = m_y - se * critval,
         upr = m_y + se * critval) %>% 
  rename(likertAgree_recoded = m_y)

# Plot
plot_fig1_inset <-
  condition_means %>% 
  mutate(condition_label = case_when(condition == "Cue-only" ~ "Cue\nonly",
                                     condition == "Info-only" ~ "Message\nonly",
                                     TRUE ~ condition)) %>% 
  mutate(condition_label = factor(condition_label, 
                                  levels = c("Control", "Cue\nonly", "Message\nonly", "Both"))) %>% 
  ggplot(aes(x = likertAgree_recoded, y = fct_rev(condition_label))) +
  labs(y = "Condition", x = "Agreement with\nIn-Party Leader (1-7)",
       title = "Raw Means [95% CI]") +
  geom_point(size = 1.5) +
  geom_errorbarh(aes(xmin = lwr, xmax = upr), height = 0) +
  geom_text(aes(x = likertAgree_recoded, y = condition_label, label = sprintf("%.2f", round(likertAgree_recoded, 2))), 
            position = position_nudge(y = 0.3)) +
  xlim(3.75, 5.25) +
  theme(plot.background = element_rect(colour = "black", fill = "white", size = 0.75),
        axis.title = element_text(size = 9))

# > Fixed effects ----

summary_pop__primary <-
  summary_pop__primary %>% 
  mutate(parameter_label = case_when(parameter == "b_cue" ~ "ATE of\nParty Leader\nCue",
                                     parameter %in% c("b_info", "b_info_under_cue") ~ "ATE of\nPersuasive\nMessage",
                                     parameter == "b_info:cue" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                     TRUE ~ parameter)) %>% 
  mutate(parameter_label = factor(parameter_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                              "ATE of\nPersuasive\nMessage",
                                                              "ATE of\nParty Leader\nCue")))

# plot_fig1 <- 
#   summary_pop__primary %>% 
#   filter(str_detect(parameter, "b_"), parameter != "b_Intercept") %>% 
#   ggplot(aes(x = parameter_label, y = value)) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.4, size = 1.5) +
#   coord_flip() +
#   labs(x = "", y = "Estimate (Median [95% HPDI])\n") +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
#   geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), 
#             vjust = -1, show.legend = F, size = 4)

plot_fig1 <-
  summary_pop__primary %>% 
  filter(str_detect(parameter, "b_"), parameter != "b_Intercept") %>% 
  mutate(group = case_when(parameter == "b_info_under_cue" ~ 1, TRUE ~ 2)) %>% 
  ggplot(aes(x = parameter_label, y = value, color = factor(group), shape = factor(group))) +
  geom_point(size = 3, position = position_dodge(.7)) +
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, size = 1, position = position_dodge(.7)) +
  coord_flip() +
  labs(x = "", y = "Estimate (Median [95% HPDI])\n") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), 
            vjust = -1, show.legend = F, size = 4) +
  theme(legend.position = "none") +
  scale_color_manual(values = c("grey40", "black")) +
  scale_shape_manual(values = c("triangle", "circle")) +
  annotate(geom = "text", label = "Countervailing\nleader cue absent", x = 2.45, y = -0.33, size = 3.5) +
  annotate(geom = "text", label = "Countervailing\nleader cue present", x = 1.65, y = -0.36, color = "grey40", size = 3.5)

g <- 
  ggdraw() +
  draw_plot(plot_fig1) +
  draw_plot(plot_fig1_inset, x = 0.6, y = 0.2, width = 0.35, height = 0.4)

ggsave(plot = g, filename = "figures/figure_1.png", height = 6, width = 8, dpi = 300)
ggsave(plot = g, filename = "figures/figure_1.pdf", height = 6, width = 8, dpi = 300)

# Files for Appendix ----

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


# > Primary model ----

# >> Traceplots ----

# List parameters
params__primary <- 
  samples_pop__primary %>% 
      select(-c("chain", "iter")) %>% 
      names()

# Make plots and save
traceplots__primary <- map(params__primary, ~fun_traceplots(param = .x, samples = samples_pop__primary))
plot_grid(plotlist = traceplots__primary)
ggsave("appendix/figures/traceplots__primary_model.png", dpi = 300, height = 12, width = 14)

# >> Summary table ----
table_primary <- readRDS("model_fitting/primary/cluster_output/tables/summary_table__primary.rds")

table_primary_mod <-
  table_primary %>% 
  mutate(Group = case_when(Group == "subject_id" ~ "Respondent",
                           Group == "item_label" ~ "Policy question",
                           TRUE ~ str_to_sentence(Group)))

# Wrangle column names
colnames(table_primary_mod) <- c("Group", "Term", "Estimate", "Est. Error",
                                 "L. 95\\% CI", "H. 95\\% CI", "Eff. Samples", "$\\hat{R}$")

saveRDS(table_primary_mod, "appendix/tables/table__primary_model.rds")


# Table for main text ----

# Wrangle table for matching with summary medians
x <-
  table_primary %>% 
  mutate(parameter = case_when(Group == "fixed" ~ paste0("b_", Term),
                               Group == "residual" ~ Term,
                               Group == "item_label" & str_detect(Term, "sd") ~ paste0("sd_item_label__", Term),
                               Group == "item_label" & str_detect(Term, "cor") ~ paste0("cor_item_label__", Term),
                               Group == "subject_id" & str_detect(Term, "sd") ~ paste0("sd_pid__", Term),
                               Group == "subject_id" & str_detect(Term, "cor") ~ paste0("cor_pid__", Term))) %>% 
  # Remove nuisance characters
  mutate(parameter = str_replace_all(parameter, c("__sd" = "__", "__cor" = "__", "\\(" = "", "\\)" = "", "," = "__")))

# Join with summary medians and wrangle
y <-
  left_join(summary_pop__primary %>% 
              select(parameter, value, .lower, .upper) %>% 
              filter(parameter != "b_info_under_cue"),
            x %>% 
              select(parameter, Eff.Sample, Rhat, Est.Error),
            by = "parameter")

y <-
  y %>% 
  mutate(group = case_when(str_detect(parameter, "b_") ~ "Fixed effects",
                           str_detect(parameter, "item_label") ~ "Random effects (policy issues)",
                           str_detect(parameter, "pid") ~ "Random effects (respondents)",
                           parameter == "sigma" ~ "Residual")) %>% 
  mutate(parameter_mod = str_replace_all(parameter, c("_item_label" = "", "_pid" = ""))) %>% 
  mutate(parameter_mod = str_replace_all(parameter_mod, c("cor__" = "Corr(", "sd__" = "SD(", "b_" = "", ":" = " x "))) %>% 
  mutate(parameter_mod = str_replace_all(parameter_mod, "__", " | ")) %>% 
  mutate(parameter_mod = case_when(str_detect(parameter_mod, "\\(") ~ paste0(parameter_mod, ")"),
                                   TRUE ~ parameter_mod)) %>% 
  mutate(parameter_mod = str_replace_all(parameter_mod, c("cue" = "Cue", "info" = "Message", "Intercept" = "Constant"))) %>% 
  mutate(order_var = case_when(group == "Fixed effects" ~ 1,
                               str_detect(parameter_mod, "SD") ~ 2,
                               str_detect(parameter_mod, "Corr") ~ 3,
                               group == "Residual" ~ 4)) %>% 
  arrange(order_var, parameter_mod) %>% 
  filter(group != "Random effects (respondents)") %>% 
  select(group, parameter_mod, value, Est.Error, .lower, .upper, Eff.Sample, Rhat) %>% 
  mutate(parameter_mod = str_replace_all(parameter_mod, "Constant", "Intercept"))

y <- 
  y %>% 
  mutate(across(c(value, Est.Error, .lower, .upper, Rhat), ~format(round(.x, 2), nsmall = 2)),
         Eff.Sample = round(Eff.Sample, 0))

names(y) <- c("Parameter group", "Parameter", "Estimate", "Std. Error", "Lower 95% HPDI", "Upper 95% HPDI", "Effective Samples", "Rhat")

# Write to file
write.csv(y, "tables/table__primary_model__main_text.csv", row.names = F)

