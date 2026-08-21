
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# ROBUSTNESS ANALYSES ####

# library(tidyverse)
# library(brms)
# library(tidybayes)
# library(cowplot)
# library(broom)
# library(estimatr)
# library(cmdstanr)

set.seed(42)

theme_set(
  theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text = element_text(color = "black"))
)

critval <- qnorm(.025, lower.tail = F)

# Read in data ----
df <- readRDS("data/data_RM.rds")

df_for_model <-
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  mutate(condition = factor(condition, levels = c("Control", "Cue-only", "Info-only", "Both")))

# Post-treatment missings ----

# Count
df_for_model %>% 
  filter(item_seen == TRUE) %>% 
  count(post_treat_missing_data)

# Count by condition
df_for_model %>% 
  filter(item_seen == TRUE) %>% 
  group_by(condition) %>% 
  summarise(n_missing = sum(post_treat_missing_data),
            n_total = n(),
            prop_missing = mean(post_treat_missing_data)) %>% 
  saveRDS("appendix/tables/robust__condition_na_count.rds")

# F-test for differences by condition
df_for_model %>% 
  filter(item_seen == TRUE) %>% 
  do(tidy(anova(lm(post_treat_missing_data ~ condition, data = .)))) %>% 
  mutate(across(c(sumsq, meansq, statistic), ~round(.x, 3))) %>% 
  #mutate(p.value = case_when(p.value < .001 ~ "<.001", TRUE ~ as.character(p.value))) %>% 
  saveRDS("appendix/tables/robust__condition_na_ftest.rds")

# Demographic summary ----

df_for_model <-
  df_for_model %>% 
  mutate(vote_prefer_trump = ifelse(vote_pref == "Trump", 1, 0))

# List of covariates
covs <- c("age_survey", "female", "white", "ba_degree", "republican", "party_strength", "conservative", "ideo_strength", "vote_prefer_trump", "PK_sum")

cov_labels <- c("age_survey" = "Age in Years", "ba_degree" = "BA Degree [0,1]", "conservative" = "Conservative [0,1]", 
                "female" = "Female [0,1]", "ideo_strength" = "Ideology Strength [0-3]", "party_strength" = "Party ID Strength [0-3]",
                "PK_sum" = "Political Knowledge Score [0-4]", "republican" = "Republican [0,1]", "white" = "White [0,1]",
                "vote_prefer_trump" = "Vote/Prefer Trump over Biden [0,1]")

# Compute and plot demographic means
out_demos <-
  df_for_model %>% 
  drop_na(likertAgree_recoded) %>% 
  distinct(pid, .keep_all = T) %>% 
  select(all_of(covs)) %>% 
  pivot_longer(everything(), names_to = "covariate") %>% 
  group_by(covariate) %>% 
  summarise(mean_y = mean(value, na.rm = T)) %>% 
  ungroup() %>% 
  mutate(covariate = str_replace_all(covariate, cov_labels)) %>% 
  rename(Covariate = covariate,
         Mean = mean_y)

saveRDS(out_demos, "appendix/tables/table__demographics.rds")

# Balance checks on covariates ----

# F-tests on covariate balance at the policy issue level
out_cov_checks <-
  map(covs,
      function(.x) {
        
        model_formula <- as.formula(paste0(.x, "~ condition"))
        
        out_all <-
          df_for_model %>% 
          drop_na(likertAgree_recoded) %>% 
          group_by(item) %>% 
          do(tidy(anova(lm(model_formula, data = .)))) %>% 
          mutate(covariate = .x) %>%
          ungroup()

        
      }) %>% 
  bind_rows() %>% 
  filter(term != "Residuals") %>% 
  mutate(covariate = str_replace_all(covariate, cov_labels))

# Plot p values
plot_ps <-
  map(unique(out_cov_checks$covariate),
      function(.x) {
        
        df_cov <-
          out_cov_checks %>%
          filter(covariate == all_of(.x))
        
        small_p <- df_cov %>% arrange(p.value) %>% slice(1) %>% pull(p.value)
        
        g <-
          df_cov %>% 
          ggplot(aes(x = p.value)) + 
          geom_histogram() +
          labs(title = .x, y = "", x = "") +
          theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
          coord_cartesian(x = c(0, 1)) +
          annotate(geom = "text", x = 0.7, y = 6, 
                   label = paste0("Smallest p-value = ", format(round(small_p, 4), nsmall = 4)))
        
      })

names(plot_ps) <- unique(out_cov_checks$covariate)

# Join and save
gp <-
  plot_grid(plot_ps$`Age in Years`,
            plot_ps$`Female [0,1]`,
            plot_ps$`White [0,1]`,
            plot_ps$`BA Degree [0,1]`,
            plot_ps$`Republican [0,1]`,
            plot_ps$`Party ID Strength [0-3]`,
            plot_ps$`Conservative [0,1]`,
            plot_ps$`Ideology Strength [0-3]`,
            plot_ps$`Political Knowledge Score [0-4]`, ncol = 3)

ggsave(plot = gp, filename = "appendix/figures/robust__balance_check__ftests.png", dpi = 300, height = 8, width = 10)

# Fit primary model minus respondent random effects ----

fit__no_respondent_rfx <- 
  brm(data = df_for_model %>% drop_na(likertAgree_recoded),
      family = gaussian,
      formula = likertAgree_recoded ~ 1 + info*cue + (1 + info*cue | item_label),
      prior = c(prior(normal(4, 1.5), class = Intercept),
                prior(normal(0, 2),   class = b),
                prior(exponential(1), class = sd),
                prior(exponential(1), class = sigma),
                prior(lkj(2),         class = cor)),
      iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42,
      file = "fits/fit__primary__no_respondent_rfx")

summary_pop__no_respondent_rfx <-
  posterior_samples(fit__no_respondent_rfx,  add_chain = TRUE) %>% 
  select(chain, iter, 1:sigma) %>% 
  pivot_longer(cols = 3:ncol(.), names_to = "parameter") %>% 
  group_by(parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup() %>% 
  mutate(Model = "Without Respondent Random Effects")

# > Plot alongside primary model results ----

# Get and summarize samples from primary model
samples_pop__primary <- readRDS("model_fitting/primary/cluster_output/samples/samples_pop__primary.rds")

summary_pop__primary <-
  samples_pop__primary %>% 
  pivot_longer(cols = 3:ncol(.), names_to = "parameter") %>% 
  group_by(parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup() %>% 
  mutate(Model = "With Respondent Random Effects (Primary Model)")

# Plot
g <-
  summary_pop__no_respondent_rfx %>% 
  bind_rows(summary_pop__primary) %>% 
  mutate(parameter_type = case_when(str_detect(parameter, "b_Intercept") ~ "Intercept",
                                    str_detect(parameter, "b_info|b_cue|b_info:cue") ~ "Key Fixed Effects",
                                    str_detect(parameter, "sd") ~ "SDs",
                                    str_detect(parameter, "cor") ~ "Correlations",
                                    str_detect(parameter, "sigma") ~ "Sigma")) %>% 
  mutate(parameter_type = factor(parameter_type, levels = c("Intercept", "Key Fixed Effects", "Correlations", "SDs", "Sigma"))) %>% 
  mutate(parameter = str_replace_all(parameter,
                                     c("item_label" = "policy_question", 
                                       "pid"        = "respondent"))) %>% 
  ggplot(aes(x = fct_rev(parameter), y = value, color = fct_rev(Model), shape = fct_rev(Model))) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red3") +
  geom_point(position = position_dodge(.5), size = 2.5) +
  geom_errorbar(aes(ymin = .lower, ymax = .upper), position = position_dodge(.5), width = 0) +
  coord_flip() +
  facet_wrap(~parameter_type, scales = "free", ncol = 2) +
  theme(legend.position = "top",
        legend.title = element_blank()) +
  scale_color_manual(values = c("grey", "black")) +
  guides(color = guide_legend(reverse = T), shape = guide_legend(reverse = T)) +
  labs(x = "Parameter", y = "Estimate (Median [95% HPDI])")

ggsave(plot = g, filename = "appendix/figures/robust__comparing_models.png", dpi = 300, height = 12, width = 10)

# Compute worst-case bounds for post-treatment missings ----

# Impute values
df_imputed <-
  df_for_model %>% 
  filter(item_seen == TRUE) %>% # post-treatment
  mutate(likertAgree_recoded_imputed = case_when(is.na(likertAgree_recoded) & condition == "Control" ~ 7,
                                                 is.na(likertAgree_recoded) & condition == "Cue-only" ~ 1,
                                                 is.na(likertAgree_recoded) & condition == "Info-only" ~ 1,
                                                 is.na(likertAgree_recoded) & condition == "Both" ~ 7,
                                                 TRUE ~ likertAgree_recoded))

# Fit
fit__imputed <- 
  brm(data = df_imputed,
      family = gaussian,
      formula = likertAgree_recoded_imputed ~ 1 + info*cue + (1 + info*cue | item_label),
      prior = c(prior(normal(4, 1.5), class = Intercept),
                prior(normal(0, 2),   class = b),
                prior(exponential(1), class = sd),
                prior(exponential(1), class = sigma),
                prior(lkj(2),         class = cor)),
      iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42,
      file = "fits/fit__primary__imputed_worst_case")

# Summarise
summary_pop__imputed <-
  posterior_samples(fit__imputed,  add_chain = TRUE) %>% 
  select(chain, iter, 1:sigma) %>% 
  pivot_longer(cols = 3:ncol(.), names_to = "parameter") %>% 
  group_by(parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup()

# Plot
summary_pop__imputed %>% 
  mutate(parameter_label = case_when(parameter == "b_cue" ~ "ATE of\nParty Leader\nCue",
                                     parameter == "b_info" ~ "ATE of\nPersuasive\nMessage",
                                     parameter == "b_info:cue" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                     TRUE ~ parameter)) %>% 
  mutate(parameter_label = factor(parameter_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                              "ATE of\nPersuasive\nMessage",
                                                              "ATE of\nParty Leader\nCue"))) %>% 
  filter(str_detect(parameter, "b_"), parameter != "b_Intercept") %>% 
  ggplot(aes(x = parameter_label, y = value)) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7) +
  coord_flip() +
  labs(x = "", y = "Estimate (Median [95% HPDI])\n") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), 
            vjust = -1, show.legend = F, size = 4)

ggsave("appendix/figures/robust__primary__imputed_worst_case.png", dpi = 300, height = 6, width = 8)

# Examine order effects ----

# Fit
list_fits_order_fx <-
  map(c(1:5),
      function(.x) {
        
          brm(data = df_for_model %>% drop_na(likertAgree_recoded) %>% filter(order_variable == .x),
              family = gaussian,
              formula = likertAgree_recoded ~ 1 + info*cue + (1 + info*cue | item_label),
              prior = c(prior(normal(4, 1.5), class = Intercept),
                        prior(normal(0, 2),   class = b),
                        prior(exponential(1), class = sd),
                        prior(exponential(1), class = sigma),
                        prior(lkj(2),         class = cor)),
              iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42,
              control = list(adapt_delta = 0.9),
              backend = "cmdstanr",
              file = paste0("fits/fit__order_fx__order__", .x))
          
        })

names(list_fits_order_fx) <- c(1:5)

# Summarise
params <- c("b_cue", "b_info", "b_info:cue")

summary_order_fx <-
  imap(list_fits_order_fx,
       function(.x, .y) {
         
         samples_pop <-
           posterior_samples(.x, pars = params, add_chain = TRUE) %>%
           select(chain, iter, everything())
         
         samples_pop %>% 
           pivot_longer(cols = 3:ncol(.), names_to = "parameter") %>% 
           group_by(parameter) %>% 
           median_hdi(value, .width = 0.95) %>% 
           ungroup() %>% 
           mutate(position = .y)
         
       }) %>% 
  bind_rows()

g <- 
  summary_order_fx %>% 
  mutate(parameter_label = case_when(parameter == "b_cue" ~ "ATE of\nParty Leader\nCue",
                                     parameter == "b_info" ~ "ATE of\nPersuasive\nMessage",
                                     parameter == "b_info:cue" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                     TRUE ~ parameter)) %>% 
  mutate(parameter_label = factor(parameter_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                              "ATE of\nPersuasive\nMessage",
                                                              "ATE of\nParty Leader\nCue"))) %>% 
  ggplot(aes(x = parameter_label, y = value, color = fct_rev(position), shape = fct_rev(position))) +
  geom_point(size = 2.5, position = position_dodge(.7)) +
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.7, position = position_dodge(.7)) +
  coord_flip() +
  labs(x = "", y = "Estimate (Median [95% HPDI])\n") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_text(aes(label = sprintf("%.2f", round(value, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4) +
  guides(color = guide_legend(reverse = T, title = "Order"), shape = guide_legend(reverse = T, title = "Order")) +
  theme(legend.position = c(0.9, 0.2),
        legend.box.background = element_rect())

ggsave(plot = g, filename = "appendix/figures/robust__order_fx.png", dpi = 300, height = 8, width = 8)

# Raw estimates ----

# Overall
df_for_model %>% 
  drop_na(likertAgree_recoded) %>% 
  do(tidy(lm_robust(likertAgree_recoded ~ 1 + cue*info, data = ., se_type = "HC3"))) %>% 
  filter(term != "(Intercept)") %>% 
  mutate(term_label = case_when(term == "cue" ~ "ATE of\nParty Leader\nCue",
                                term == "info" ~ "ATE of\nPersuasive\nMessage",
                                term == "cue:info" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                TRUE ~ term)) %>% 
  mutate(term_label = factor(term_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                    "ATE of\nPersuasive\nMessage",
                                                    "ATE of\nParty Leader\nCue"))) %>% 
  ggplot(aes(x = term_label, y = estimate)) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0, alpha = 0.7) +
  coord_flip() +
  labs(x = "", y = "Estimate [95% CI]\n") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_text(aes(label = sprintf("%.2f", round(estimate, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4)

ggsave("appendix/figures/raw_estimates__overall.png", dpi = 300, height = 6, width = 8)

# Disaggregated by policy issue
df_for_model %>% 
  drop_na(likertAgree_recoded) %>% 
  group_by(item_label) %>% 
  do(tidy(lm_robust(likertAgree_recoded ~ 1 + cue*info, data = ., se_type = "HC3"))) %>% 
  filter(term != "(Intercept)") %>% 
  mutate(term_label = case_when(term == "cue" ~ "ATE of\nParty Leader\nCue",
                                term == "info" ~ "ATE of\nPersuasive\nMessage",
                                term == "cue:info" ~ "Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                TRUE ~ term)) %>% 
  mutate(term_label = factor(term_label, levels = c("Change in\nPersuasive\nMessage ATE\nUnder Party\nLeader Cue",
                                                    "ATE of\nPersuasive\nMessage",
                                                    "ATE of\nParty Leader\nCue"))) %>% 
  ggplot(aes(x = term_label, y = estimate)) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0, alpha = 0.7) +
  coord_flip() +
  facet_wrap(~item_label, ncol = 4) +
  labs(x = "", y = "Estimate [95% CI]\n") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_text(aes(label = sprintf("%.2f", round(estimate, 2))), position = position_dodge(.7), vjust = -1, show.legend = F, size = 4)

ggsave("appendix/figures/raw_estimates__by_policy_issue.png", dpi = 300, height = 14, width = 12)
