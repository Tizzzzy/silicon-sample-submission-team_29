
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

#### MODEL FITTING FILE: Interaction models ####

# library(tidyverse)
# library(brms)
set.seed(42)

# Read in data ----
#df <- readRDS("data_prepared.rds")

# Wrangle ----

df_for_model <-
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  drop_na(likertAgree_recoded) %>% 
  
  # Wrangle relevant moderators
  mutate(trump_republican = case_when(vote_party == "Trump-Republican" ~ 1,
                                      vote_party == "Biden-Democrat" ~ 0),
         two_sided_cue    = case_when(cue_type == "two_sided" ~ 1,
                                      cue_type == "one_sided" ~ 0)) %>% 
  mutate(PK_sum_z           = as.numeric(scale(PK_sum)),
         age_z              = as.numeric(scale(age_survey)),
         trump_republican_c = as.numeric(scale(trump_republican, scale = F, center = T)),
         ba_degree_c        = as.numeric(scale(ba_degree, scale = F, center = T)),
         strong_partisan_c  = as.numeric(scale(strong_partisan, scale = F, center = T)),
         female_c           = as.numeric(scale(female, scale = F, center = T)),
         two_sided_cue_c    = as.numeric(scale(two_sided_cue, scale = F, center = T)))

# Fit models ----

list_moderators <- list("trump_republican_c",
                        "ba_degree_c",
                        "PK_sum_z",
                        "strong_partisan_c",
                        "age_z",
                        "female_c",
                        "two_sided_cue_c")

# Write function to fit model and write samples to file for each moderator ----
fun_fit_model <- function(data = NULL, mod_var = NULL) {
  
  # Rename moderator
  df_new <-
    data %>% 
    rename(mod = mod_var) 
  
  # Fit ----
  fit <- 
    brm(data = df_new,
        family = gaussian,
        formula = likertAgree_recoded ~ 1 + info*cue*mod + (1 + info*cue*mod | item_label),
        prior = c(prior(normal(4, 1.5), class = Intercept),
                  prior(normal(0, 2),   class = b),
                  prior(exponential(1), class = sd),
                  prior(exponential(1), class = sigma),
                  prior(lkj(2),         class = cor)),
        iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42,
        #control = list(adapt_delta = 0.9),
        file = paste0("fit__interaction_model__", mod_var))
  
  # Get summary table ----
  
  fit_summary <- summary(fit)
  
  summary_table <-
    bind_rows(
      as_tibble(fit_summary$fixed)             %>% mutate(Term = row.names(fit_summary$fixed),             Group = "fixed"),
      as_tibble(fit_summary$spec_pars)         %>% mutate(Term = row.names(fit_summary$spec_pars),         Group = "residual"),
      as_tibble(fit_summary$random$item_label) %>% mutate(Term = row.names(fit_summary$random$item_label), Group = "item_label")
    ) %>%
    select(Group, Term, everything()) %>% 
    mutate(moderator = mod_var)
  
  saveRDS(summary_table, paste0("summary_table__interaction_model__", mod_var, ".rds")) # Write to file
  
  # Extract posterior samples ----
  
  # > Population-level samples ----
  params <- parnames(fit)[1:nrow(summary_table)]
  
  samples_pop <-
    posterior_samples(fit,
                      pars = params,
                      add_chain = TRUE) %>%
    select(chain, iter, everything())
  
  saveRDS(samples_pop, paste0("samples_pop__interaction_model__", mod_var, ".rds")) # Write
  
  # Record divergent transitions ----
  divergent  <- nuts_params(fit) %>% filter(str_detect(Parameter, "divergent"), Value == 1) %>% nrow()
  capture.output(paste0("divergent transitions = ", divergent), file = paste0("divergences__interaction_model__", mod_var, ".txt"))
  
}


# Fit
map(list_moderators,
    ~fun_fit_model(data = df_for_model, mod_var = .x))


