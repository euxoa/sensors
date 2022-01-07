library(dplyr)
library(ggplot2)

zna <- function (x) ifelse(is.na(x), 0, x)

ewma <- function (x, ratio) stats::filter(x * ratio, 1 - ratio, "recursive", init = x[1])

# np.array((time.time(), temp, pressure, humidity, lux, proximity) + x_gas + x_pms
clr_tbl <- data.frame(
  idx = c(1, 7, 8, 9, 4, 5, 10, 11, 12, 3, 6, 2, 13, 14),
  name = c("time", "G_nh3", "G_ox", "G_red", "hum", "light", 
           "pm010", "pm025", "pm100", "pressure", "prox", "temp", "zfast", "z"), 
  clr = c("white", "green", "red", "blue", "black", "black", 
          "red", "green", "blue", "black", "black", "black", "red", "blue"),
  name2 = c("t", "G", "G", "G", "hum", "light", "pm", "pm", "pm", "pressure", "prox", "temp", "z", "z")) %>% 
  arrange(idx)

get_data <- function() {
  system("rsync -vz pi@100.88.133.109:sensors/3.txt .")
  readr::read_table("3.txt", col_names=F, col_types="nnnnnnnnnnnnnn") %>%
  setNames(clr_tbl$name) %>% mutate(time=lubridate::as_datetime(time, tz="EET")) }

graph <- function (d, t0="2021-01-07 00:00:00")  
  d %>% mutate(
         light = log10(light+.01), prox = log10(prox),
         #z = log10(z), zfast = log10(zfast), 
         G_nh3 = G_nh3 - mean(G_nh3), G_red = G_red - mean(G_red), G_ox = G_ox - mean(G_ox)) %>%
         #light=ewma(light, .1),
         #pm010 = ewma(pm010, .05), pm025 = ewma(pm025, .05), pm100 = ewma(pm100, .05)) %>%
  filter(row_number()>-1 & time > lubridate::as_datetime(t0, tz="EET")) %>%
  tidyr::pivot_longer(!time) %>%
  left_join(clr_tbl) %>%
    ggplot(aes(x=time, y=value, color=I(clr))) + geom_line(alpha=.5, size=.3) +
      facet_grid(name2 ~ ., scales="free_y") + theme_minimal()


