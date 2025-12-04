library(tidyverse)
library(dplyr)
library(readr)
library(GGally)

output_data <- list.files(path="data/", full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_rows %>% tibble()

output_data <- output_data %>% 
  mutate(
    viable = ifelse(cost == Inf, 0, 1)
  )

output_data |> 
  summarise(
    mean(viable),
    mean(M),
    range(M),
    mean(U),
    range(U),
    mean(Middle),
    range(Middle),
    mean(Sorta),
    range(Sorta),
    range(cost)
  )

output_data |> filter(U > M) |> summarise(min(U))


interesting_points <- read_csv("MoreInteresting.csv", col_names = FALSE)
interesting_points <- interesting_points |> 
  rename(
    r_hm = "X1", r_hm_h = "X2",
    r_uh = "X3", r_uh_h = "X4",
    r_mh = "X5", r_mh_h = "X6",
    r_hu = "X7", r_hu_h = "X8",
    cost = "X9"
  ) |> 
  mutate(
    cat_cost = ifelse(cost > 0.6, "high", ifelse(cost > 0.3, "medium", "low")),
    cat_cost = fct_relevel(cat_cost, "high", "medium", "low")
  )

plot <- ggpairs(
  interesting_points, columns = 1:8,
  aes(color=cat_cost),upper = list(continuous = "blankDiag")
  ) +
  scale_color_viridis_d() + 
  theme_dark()

bounds <- c(300, 6, 0.04, 0.0008, 0.25, 0.005, 0.35, 0.007)


# Loop to set common x and y axis limits for all scatter plots (lower triangle)
for(i in 1:plot$nrow) {
  for(j in 1:(i-1)) {
    plot[i,j] <- plot[i,j] +
      scale_x_continuous(limits = c(0, bounds[j])) +
      scale_y_continuous(limits = c(0, bounds[i]))
  }
}

plot + scale_color_viridis_d() + theme_dark()
