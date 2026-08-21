
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# Estimate issue-level treatment effects ####

# library(tidyverse)
# library(brms)
# library(tidybayes)
# library(estimatr)
# library(cowplot)
# library(broom)
# library(cmdstanr)
# library(MASS, exclude = "select")

set.seed(42)

theme_set(
  theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text = element_text(color = "black"))
)

# Read in samples ----
samples_pop__primary  <- readRDS("model_fitting/primary/cluster_output/samples/samples_pop__primary.rds")
samples_item__primary <- readRDS("model_fitting/primary/cluster_output/samples/samples_item__primary.rds")

# Wrangle
summary_item__primary <-
  samples_item__primary %>%
  pivot_longer(cols = 1:ncol(.),
               names_pattern = "(.+)[.](.+$)",
               names_to = c("item_label", ".value")) %>% 
  mutate(info_under_cue = info + `info:cue`) %>% 
  pivot_longer(cols = 2:ncol(.),
               names_to = "parameter") %>% 
  group_by(item_label, parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup()

summary_pop__primary <-
  samples_pop__primary %>% 
  mutate(cate_under_cue = b_info + `b_info:cue`) %>% 
  select(b_info, cate_under_cue, `b_info:cue`) %>% 
  pivot_longer(cols = everything(),
               names_to = "parameter") %>% 
  group_by(parameter) %>% 
  median_hdi(value, .width = 0.95) %>% 
  ungroup()

# Plot observed issue-level effects (regularized) ----

# Fix mispelling
summary_item__primary <-
  summary_item__primary %>% 
  mutate(item_label = str_replace_all(item_label, "Saudia", "Saudi"))
  

plot_int_fx <-
  summary_item__primary %>%
  filter(str_detect(parameter, ":")) %>% 
  ggplot(aes(x = fct_rev(item_label), y = value)) +
  coord_flip() +
  # Add average effect
  geom_hline(yintercept = summary_pop__primary %>% filter(parameter == "b_info:cue") %>% pull(value), color = "black", size = 3, alpha = 0.5) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, size = 2, alpha = 0.15) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(x = "",
       y = "Interaction Estimate (Median [95% HPDI])",
       title = "\nInteraction Effects") +
  theme(panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank())
  

plot_cates <- 
  summary_item__primary %>%
  filter(parameter %in% c("info", "info_under_cue")) %>% 
  ggplot(aes(x = fct_rev(item_label), y = value, color = parameter, shape = parameter)) +
  coord_flip() + 
  # Add average effects
  geom_hline(yintercept = summary_pop__primary %>% filter(parameter == "b_info") %>% pull(value), color = "green4", size = 3, alpha = 0.5) +
  geom_hline(yintercept = summary_pop__primary %>% filter(parameter == "cate_under_cue") %>% pull(value), color = "orange3", size = 3, alpha = 0.5) +
  geom_point(size = 2.5) + 
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0, alpha = 0.15, size = 2) + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_color_manual(values = c("green4", "orange3"),
                     labels = c("Absent", "Present"),
                     name = "Countervailing\nLeader Cue:") +
  scale_shape_manual(values = c("circle", "triangle"),
                     labels = c("Absent", "Present"),
                     name = "Countervailing\nLeader Cue:") +
  labs(x = "Policy Issue", 
       y = "Conditional Average Treatment Effect (Median [95% HPDI])",
       title = "Conditional Average\nTreatment Effects of Persuasive Messaging") +
  theme(panel.grid.minor = element_blank(), 
        panel.grid.major.y = element_blank(),
        legend.position = c(0.15, 0.4),
        #legend.title = element_blank(),
        legend.box.background = element_rect())

g <- 
  plot_grid(plot_int_fx,
          plot_cates + 
            theme(axis.text.y = element_blank(), 
                  axis.ticks.y = element_blank()) +
            labs(x = ""), 
          ncol = 2,
          labels = "AUTO")

ggsave(plot = g, filename = "figures/figure_3.png", height = 8, width = 12, dpi = 300)
ggsave(plot = g, filename = "figures/figure_3.pdf", height = 8, width = 12, dpi = 300)

# Simulate and plot hypothetical issue-level effects ----

# > Write function to simulate political issues ----
fun_simulate_issues <- function(samples = NULL, n_sim = NULL, n_issues = NULL) {
  
  # Draw one sample from the posterior
  if(n_sims == nrow(samples)) { draw_post <- samples %>% slice(n_sim) } # if n_sims equals the full posterior samples, draw them rowwise
  if(n_sims != nrow(samples)) { 
    set.seed(n_sim)
    draw_post <- samples %>% sample_n(1) 
  } # if n_sims is not equal to the full posterior samples, draw them randomly with replacement
  
  # Store population averages
  a  <- draw_post %>% pull(b_Intercept) 
  b1 <- draw_post %>% pull(b_cue) 
  b2 <- draw_post %>% pull(b_info) 
  b3 <- draw_post %>% pull(`b_info:cue`) 
  
  # Store standard deviations
  sigma_a  <- draw_post %>% pull(sd_item_label__Intercept)    
  sigma_b1 <- draw_post %>% pull(sd_item_label__cue)    
  sigma_b2 <- draw_post %>% pull(sd_item_label__info)      
  sigma_b3 <- draw_post %>% pull(`sd_item_label__info:cue`)      
  
  # Store correlations
  rho__a_b1  <- draw_post %>% pull(cor_item_label__Intercept__cue) 
  rho__a_b2  <- draw_post %>% pull(cor_item_label__Intercept__info) 
  rho__a_b3  <- draw_post %>% pull(`cor_item_label__Intercept__info:cue`) 
  rho__b1_b2 <- draw_post %>% pull(cor_item_label__info__cue) 
  rho__b1_b3 <- draw_post %>% pull(`cor_item_label__cue__info:cue`)
  rho__b2_b3 <- draw_post %>% pull(`cor_item_label__info__info:cue`) 
  
  # Compute covariances for covariance matrix
  cov__a_b1  <- sigma_a  * sigma_b1 * rho__a_b1
  cov__a_b2  <- sigma_a  * sigma_b2 * rho__a_b2
  cov__a_b3  <- sigma_a  * sigma_b3 * rho__a_b3
  cov__b1_b2 <- sigma_b1 * sigma_b2 * rho__b1_b2
  cov__b1_b3 <- sigma_b1 * sigma_b3 * rho__b1_b3
  cov__b2_b3 <- sigma_b2 * sigma_b3 * rho__b2_b3
  
  # Create covariance matrix
  sigma  <- matrix(c(sigma_a^2, cov__a_b1,  cov__a_b2,  cov__a_b3,
                     cov__a_b1, sigma_b1^2, cov__b1_b2, cov__b1_b3,
                     cov__a_b2, cov__b1_b2, sigma_b2^2, cov__b2_b3,
                     cov__a_b3, cov__b1_b3, cov__b2_b3, sigma_b3^2), 
                   ncol = 4)
  
  # Store population means
  mu <- c(a, b1, b2, b3)
  
  # Simulate issues from the population
  set.seed(n_sim)
  
  if(n_issues == 1) { # need this alternative code for merging to df when n_issues = 1
    
    df <- MASS::mvrnorm(n = n_issues, mu, sigma) %>% 
      purrr::map2(., seq_along(.), 
                  ~data.frame(.x) %>% set_names(.y)) %>% 
      bind_cols()
    
  } else {
    
    df <- MASS::mvrnorm(n = n_issues, mu, sigma) %>% 
      data.frame()
    
  }
  
  df %>% 
    set_names("a_issue", "b1_issue", "b2_issue", "b3_issue") %>% 
    mutate(n_issue = row_number(),
           n_sim   = n_sim,
           a_mean  = a,
           b1_mean = b1,
           b2_mean = b2,
           b3_mean = b3)
  
}

n_sims           <- 8000 # draws from the posterior
n_issues_per_sim <- 1000 # number of issues per draw

# > Simulate ----
df_simulated__issues <-
  purrr::map(1:n_sims,
             ~fun_simulate_issues(samples = samples_pop__primary,
                                  n_sim = .x,
                                  n_issues = n_issues_per_sim)) %>% 
  bind_rows()

# Wrangle
df_simulated__issues <-
  df_simulated__issues %>% 
  mutate(ate_under_cue = b2_issue + b3_issue,
         ate_no_cue = b2_issue)

# Summarise
summary_simulated__issues <-
  df_simulated__issues %>% 
  group_by(n_sim) %>% 
  arrange(desc(b3_issue)) %>% 
  mutate(issue_rank = row_number()) %>% 
  ungroup() %>% 
  pivot_longer(cols = c("b3_issue", "ate_no_cue", "ate_under_cue"),
               names_to = "estimand") %>% 
  group_by(estimand, issue_rank) %>% 
  mean_qi(value, .width = c(0.66, 0.95)) %>% 
  pivot_wider(names_from = ".width",
              values_from = c(".lower", ".upper")) %>% 
  ungroup() %>% 
  mutate(n_sims = n_sims,
         n_issues_per_sim = n_issues_per_sim)

# > Plot ----
g <-
  plot_grid(
  
  summary_simulated__issues %>%
    filter(estimand == "b3_issue") %>%
    arrange(desc(issue_rank)) %>%
    mutate(issue_rank_flip = row_number()) %>%
    ggplot(aes(x = issue_rank_flip, y = value)) +
    coord_flip(ylim = c(-0.5, 0.5)) +
    geom_line(size = 1) +
    geom_ribbon(aes(ymin = .lower_0.95, ymax = .upper_0.95), fill = "black", alpha = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(x = "Policy issue",
         y = "",
         title = "Estimated\nDistribution of Interaction Effects") +
    theme(panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) +
    geom_vline(xintercept = n_issues_per_sim*0.975) +
    annotate(geom = "text", label = "Largest 2.5%", y = 0.4, x = n_issues_per_sim*0.975 + n_issues_per_sim*0.025) +
    geom_vline(xintercept = n_issues_per_sim*0.025) +
    annotate(geom = "text", label = "Smallest 2.5%", y = 0.4, x = n_issues_per_sim*0.025 - n_issues_per_sim*0.025) +
    geom_vline(xintercept = n_issues_per_sim*0.5) +
    annotate(geom = "text", label = "Mean of distribution", y = 0.35, x = n_issues_per_sim*0.5 + n_issues_per_sim*0.025),
  
  summary_simulated__issues %>%
    filter(estimand != "b3_issue") %>%
    group_by(estimand) %>%
    arrange(desc(issue_rank)) %>%
    mutate(issue_rank_flip = row_number()) %>%
    ungroup() %>%
    ggplot(aes(x = issue_rank_flip, y = value, fill = estimand)) +
    geom_line(size = 1, aes(color = estimand)) +
    coord_flip(ylim = c(-1, 0.3)) +
    geom_ribbon(aes(ymin = .lower_0.95, ymax = .upper_0.95), alpha = 0.15) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_fill_manual(values = c("green4", "orange3")) +
    scale_color_manual(values = c("green4", "orange3")) +
    labs(x = "",
         y = "",
         title = "Estimated\nDistribution of Persuasive Messaging Effects") +
    theme(panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          legend.position = "none") +
    annotate(geom = "text", label = "Countervailing\nLeader Cue\nAbsent", color = "green4",  x = n_issues_per_sim*0.85, y = -0.8) +
    annotate(geom = "text", label = "Countervailing\nLeader Cue\nPresent", color = "orange3", x = n_issues_per_sim*0.85, y = 0.15) +
    geom_vline(xintercept = n_issues_per_sim*0.975) +
    geom_vline(xintercept = n_issues_per_sim*0.025) +
    geom_vline(xintercept = n_issues_per_sim*0.5),

  ncol = 2, labels = "AUTO")

ggsave(plot = g, filename = "figures/figure_4.png", height = 8, width = 12, dpi = 300)
ggsave(plot = g, filename = "figures/figure_4.pdf", height = 8, width = 12, dpi = 300)


# Table for main text ----

x <- 
  summary_item__primary %>% 
  filter(str_detect(parameter, "info")) %>% 
  mutate(parameter_mod = case_when(parameter == "info" ~ "Message ATE (cue absent)",
                                   parameter == "info_under_cue" ~ "Message ATE (cue present)",
                                   parameter == "info:cue" ~ "Interaction")) %>% 
  select(item_label, parameter_mod, value, .lower, .upper) %>% 
  mutate(across(c(value, .lower, .upper), ~format(round(.x, 2), nsmall = 2)))

# Pivot wider
x_wide <-
  x %>% 
  mutate(est = paste0(value, " [", .lower, " | ", .upper, "]")) %>% 
  pivot_wider(id_cols = item_label, names_from = parameter_mod, values_from = est)

names(x_wide)[1] <- "Policy issue"

# Write to file
write.csv(x_wide, "tables/table__issue_level_results__main_text.csv", row.names = F)
