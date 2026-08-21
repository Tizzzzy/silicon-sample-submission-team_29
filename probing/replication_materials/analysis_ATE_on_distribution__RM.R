
# Replication code for
# Partisans’ Receptivity to Persuasive Messaging is Undiminished by Countervailing Party Leader Cues

# Estimating treatment effects on the distribution ####

# library(tidyverse)
# library(cowplot)

set.seed(42)

theme_set(
  theme_bw() + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text = element_text(color = "black"))
)

# Read in data ----

df <- readRDS("data/data_RM.rds")

df_for_model <- 
  df %>% 
  filter(vote_party %in% c("Biden-Democrat", "Trump-Republican")) %>% 
  drop_na(likertAgree_recoded)

# Wrangle data ----

# Compute proportions
y <- 
  df_for_model %>% 
  group_by(condition, likertAgree_recoded) %>% 
  summarise(n = n()) %>% 
  ungroup() %>% 
  group_by(condition) %>% 
  mutate(prop = n/sum(n)) %>% 
  ungroup() %>% 
  mutate(cue = case_when(str_detect(condition, "Both|Cue") ~ "Countervailing Leader Cue Present", TRUE ~ "Countervailing Leader Cue Absent"),
         info = case_when(str_detect(condition, "Both|Info") ~ "Persuasive Message", TRUE ~ "No Message")) %>% 
  dplyr::select(-n) %>% 
  mutate(type = "Raw Distribution")

# Wrangle wide format
y_wide <-
  y %>% 
  pivot_wider(id_cols = c("likertAgree_recoded", "cue"),
              names_from = "info",
              values_from = "prop") %>% 
  mutate(difference = `Persuasive Message` - `No Message`) %>% 
  rename(prop = difference) %>% 
  dplyr::select(-c(`Persuasive Message`, `No Message`)) %>% 
  mutate(type = "Difference")

# Wrangle plot labels
df_labels <-
  data.frame(type = "Difference",
             info = "Difference",
             cue = c("Countervailing Leader Cue Present", "Countervailing Leader Cue Absent"),
             label = "Disagreement\nwith in-party leader") %>% 
  mutate(type = fct_rev(type))

# Plot ----
g <- 
  bind_rows(y %>% mutate(type = "Raw Distribution"), 
          y_wide %>% mutate(type = "Difference", info = "Difference")) %>% 
  mutate(type = fct_rev(type),
         info = factor(info, levels = c("No Message", "Persuasive Message", "Difference"))) %>% 
  ggplot(aes(x = likertAgree_recoded, y = prop, fill = info)) +
  geom_col(position = position_dodge(.7), alpha = 0.8) +
  facet_grid(type~cue) +
  scale_x_continuous(breaks = 1:7) +
  scale_fill_manual(values = c("green4", "orange3", "grey")) +
  theme(legend.position = "top",
        legend.title = element_blank(),
        legend.box.background = element_rect(),
        panel.grid.minor.x = element_blank(),
        strip.background = element_rect(fill = "white",colour = "white"),
        strip.text = element_text(face = "bold", size = 12)) +
  labs(x = "\nAgreement with in-party leader's position\n(1 = strong disagreement, 7 = strong agreement)", 
       y = "Proportion") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_text(data = df_labels, x = 2, y = 0.12, aes(label = label)) +
  geom_segment(data = df_labels, aes(x = 0.5, xend = 3.5, y = 0.08, yend = 0.08))

ggsave(plot = g, filename = "figures/figure_2.png", height = 8, width = 8, dpi = 300)
ggsave(plot = g, filename = "figures/figure_2.pdf", height = 8, width = 8, dpi = 300)
