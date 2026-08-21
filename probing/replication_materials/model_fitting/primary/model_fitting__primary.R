
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

#### MODEL FITTING FILE ####

# library(brms)
# library(tidyverse)
set.seed(42)

# Read in data ----
#df <- readRDS("data_prepared.rds")

# Fit model ----

df_for_model <-
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  drop_na(likertAgree_recoded) %>% 
  select(pid, item, item_label, info, cue, likertAgree_recoded)

fit <- 
  brm(data = df_for_model,
      family = gaussian,
      formula = likertAgree_recoded ~ 1 + info*cue + 
        (1 + info*cue | item_label) +
        (1 + info*cue | pid),
      prior = c(prior(normal(4, 1.5), class = Intercept),
                prior(normal(0, 2),   class = b),
                prior(exponential(1), class = sd),
                prior(exponential(1), class = sigma),
                prior(lkj(2),         class = cor)),
      iter = 3000, warmup = 1000, chains = 4, cores = 4, seed = 42,
      control = list(adapt_delta = 0.99),
      file = "fit__primary")

fit_summary <- summary(fit)

summary_table <-
  bind_rows(
    as_tibble(fit_summary$fixed)             %>% mutate(Term = row.names(fit_summary$fixed),             Group = "fixed"),
    as_tibble(fit_summary$spec_pars)         %>% mutate(Term = row.names(fit_summary$spec_pars),         Group = "residual"),
    as_tibble(fit_summary$random$item_label) %>% mutate(Term = row.names(fit_summary$random$item_label), Group = "item_label"),
    as_tibble(fit_summary$random$pid)        %>% mutate(Term = row.names(fit_summary$random$pid),        Group = "subject_id")
  ) %>%
  select(Group, Term, everything())

saveRDS(summary_table, "summary_table__primary.rds") # Write to file

# Extract posterior samples ----

# > Population-level samples ----
params <- parnames(fit)[1:nrow(summary_table)]

samples_pop <-
  posterior_samples(fit,
                    pars = params,
                    add_chain = TRUE) %>%
  select(chain, iter, everything())

saveRDS(samples_pop, "samples_pop__primary.rds") # Write

# > Issue-level samples ----
x <- coef(fit, summary = FALSE)
samples_item <- as.data.frame(x$item_label)

saveRDS(samples_item, "samples_item__primary.rds") # Write

# Record divergent transitions ----
divergent <- nuts_params(fit) %>% filter(str_detect(Parameter, "divergent"), Value == 1) %>% nrow()
capture.output(paste0("divergent transitions = ", divergent), file = "divergences__primary.txt")


