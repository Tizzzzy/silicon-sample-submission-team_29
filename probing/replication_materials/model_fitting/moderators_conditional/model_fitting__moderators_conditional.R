
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

#### MODEL FITTING FILE: Conditional average effects ####

# library(brms)
# library(tidyverse)
set.seed(42)

# Read in data ----
#df <- readRDS("data_prepared.rds")

# Wrangle ----

df_for_model <-
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  drop_na(likertAgree_recoded)

# Compute some quantiles
df_for_model <-
  df_for_model %>% 
  mutate(PK_sum_tertiles = ntile(PK_sum, 3),
         age_tertiles    = ntile(age_survey, 3))

# List moderator levels
list_moderator_levels <- list("vote_party"      = c("Biden-Democrat", "Trump-Republican"),
                              "ba_degree"       = c(0, 1),
                              "PK_sum_tertiles" = c(1, 3),
                              "strong_partisan" = c(0, 1),
                              "age_tertiles"    = c(1, 3),
                              "female"          = c(0, 1),
                              "cue_type"        = c("one_sided", "two_sided"))

# Write function to output summary table and samples ----
fun_get_tables_samples <- function(fit = NULL, mod = NULL, mod_level = NULL) {
  
  # Table
  fit_summary <- summary(fit)
  
  summary_table <-
    bind_rows(
      as_tibble(fit_summary$fixed)             %>% mutate(Term = row.names(fit_summary$fixed),             Group = "fixed"),
      as_tibble(fit_summary$spec_pars)         %>% mutate(Term = row.names(fit_summary$spec_pars),         Group = "residual"),
      as_tibble(fit_summary$random$item_label) %>% mutate(Term = row.names(fit_summary$random$item_label), Group = "item_label")
    ) %>%
    select(Group, Term, everything()) %>% 
    mutate(moderator = mod,
           moderator_level = mod_level)
  
  saveRDS(summary_table, paste0("summary_table__conditional_fx__", mod, "__", mod_level, ".rds")) # Write to file
  
  # Samples
  params <- parnames(fit)[1:nrow(summary_table)]
  
  samples_pop <-
    posterior_samples(fit,
                      pars = params,
                      add_chain = TRUE) %>%
    select(chain, iter, everything())
  
  saveRDS(samples_pop, paste0("samples_pop__conditional_fx__", mod, "__", mod_level, ".rds")) # Write
  
}

# Fit ----
fits_conditional <-
  imap(list_moderator_levels,
       function(.x, .y) {
         
         df1 <- df_for_model %>% filter(get(.y) == .x[[1]])
         df2 <- df_for_model %>% filter(get(.y) == .x[[2]])
         
         file_path1 <- paste0(.y, "__", .x[[1]])
         file_path2 <- paste0(.y, "__", .x[[2]])
         
         prior_set <- c(prior_string("normal(4, 1.5)", class = "Intercept"),
                        prior_string("normal(0, 2)",   class = "b"),
                        prior_string("exponential(1)", class = "sd"),
                        prior_string("exponential(1)", class = "sigma"),
                        prior_string("lkj(2)",         class = "cor"))
         
         fit1 <- 
           brm(data = df1,
               family = gaussian,
               formula = likertAgree_recoded ~ 1 + info*cue + (1 + info*cue | item_label),
               prior = prior_set,
               iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42, control = list(adapt_delta = 0.95),
               file = paste0("fit__", file_path1))
         
         fun_get_tables_samples(fit = fit1, mod = .y, mod_level = .x[[1]])
         
         # Record divergent transitions
         divergent1 <- nuts_params(fit1) %>% filter(str_detect(Parameter, "divergent"), Value == 1) %>% nrow()
         capture.output(paste0("divergent transitions = ", divergent1), file = paste0("divergences__", file_path1, ".txt"))
         
         fit2 <- 
           brm(data = df2,
               family = gaussian,
               formula = likertAgree_recoded ~ 1 + info*cue + (1 + info*cue | item_label),
               prior = prior_set,
               iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42, control = list(adapt_delta = 0.95),
               file = paste0("fit__", file_path2))
         
         fun_get_tables_samples(fit = fit2, mod = .y, mod_level = .x[[2]])
         
         # Record divergent transitions
         divergent2 <- nuts_params(fit2) %>% filter(str_detect(Parameter, "divergent"), Value == 1) %>% nrow()
         capture.output(paste0("divergent transitions = ", divergent2), file = paste0("divergences__", file_path2, ".txt"))
         
       })

