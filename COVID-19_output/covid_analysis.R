# =============================================================================
#  COVID-19 and exchange rates: Analysis A (the whole COVID period)
#
#  Questions
#    (1) Over the COVID period as a whole, can the models forecast better for one
#        currency pair than for the others?
#    (2) Do Holt / Holt-Winters/ ARIMA / SARIMA forecast better than the random walk?

#   Structure:
#    0. Packages and settings
#    1. Data: load, align the three series, NA report
#    2. One overview figure with the COVID events marked
#    3. Forecasting toolbox: six models, error metrics, rolling-origin CV
#    4. Analysis A: whole COVID period
#    5. Which pair is most predictable?
#    6. Fitted parameters per model and pair
#    7. Export


# =============================================================================
# 0. PACKAGES AND SETTINGS
# =============================================================================
# install.packages(c("xts", "zoo", "forecast", "ggplot2", "dplyr", "tidyr"))

library(xts)
library(zoo)
library(forecast)
library(ggplot2)
library(dplyr)
library(tidyr)

# ---- Periods 
# The models need about one year of data before the first forecast.
# So 2019 is used as the initial training year, and forecasts are evaluated from
# the start of 2020 until the WHO ended the COVID emergency (5 May 2023).
TRAIN_START <- as.Date("2019-01-01")   # start of the initial training data
EVAL_START  <- as.Date("2020-01-01")   # first forecasted day is the first trading day of 2020
COVID_END   <- as.Date("2023-05-05")   # WHO declares the end of the COVID emergency

# ---- Events marked in the overview figure -----------------------------------
# type = "COVID" for COVID events, "Other" for big non-COVID shocks in the same period.
# The "Other" ones matter for the interpretation: errors in 2022 are not only COVID.
covid_events <- data.frame(
  id    = 1:8,
  date  = as.Date(c("2020-01-30", "2020-03-11", "2020-03-18", "2020-11-09",
                    "2020-12-31", "2022-02-24", "2022-09-23", "2023-05-05")),
  label = c("WHO declares global health emergency",
            "WHO declares a pandemic",
            "ECB launches emergency bond buying (PEPP)",
            "First vaccine efficacy results (Pfizer/BioNTech)",
            "Brexit transition period ends",
            "Russia invades Ukraine",
            "UK 'mini-budget'",
            "WHO ends global health emergency"),
  type  = c("COVID", "COVID", "COVID", "COVID", "Other", "Other", "Other", "COVID")
)

#  Models 
SEAS_PERIOD <- 5   # 5-day trading week
MODELS <- c("naive",         # random walk benchmark: tomorrow = today
            "ses",           # simple exponential smoothing, ETS(A,N,N)
            "holt",          # Holt's linear trend, ETS(A,A,N)
            "holt_winters",  # Holt-Winters additive, ETS(A,A,A) with period 5
            "arima",         # auto.arima, non-seasonal
            "sarima")        # auto.arima, allowed to add seasonal terms with period 5

# CV (expanding window, rolling origin)
CV_H    <- 20   # each fold forecasts 20 trading days (~1 month) ahead
CV_STEP <- 20   # a new origin every 20 trading days, so test windows do not overlap

# Output and plot style 
OUT_DIR <- "C:\\Users\\e_cic\\Desktop\\IS_seminar\\COVID-19_output"
dir.create(OUT_DIR, showWarnings = FALSE)

PAIR_COLS   <- c(gbp_eur = "#0072B2", eur_usd = "#D55E00", usd_gbp = "#009E73")
PAIR_LABELS <- c(gbp_eur = "GBP per 1 EUR (ECB)",
                 eur_usd = "USD per 1 EUR (ECB)",
                 usd_gbp = "USD per 1 GBP (FRED)")

theme_paper <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position  = "bottom",
        strip.text       = element_text(face = "bold"),
        plot.caption     = element_text(hjust = 0, size = 8, colour = "grey30"))

save_plot <- function(p, file, w = 10, h = 6) {
  ggsave(file.path(OUT_DIR, file), p, width = w, height = h, dpi = 300, bg = "white")
  print(p)
  invisible(p)
}
save_table <- function(df, file) {
  write.csv(df, file.path(OUT_DIR, file), row.names = FALSE)
  invisible(df)
}


# =============================================================================
# 1. DATA
# =============================================================================
# Reuses gbp_eur_xts, eur_usd_xts and usd_gbp_xts 
if (!all(sapply(c("gbp_eur_xts", "eur_usd_xts", "usd_gbp_xts"), exists))) {
  message("Downloading data from the ECB and FRED ...")
  library(ecb)
  library(quantmod)
  eur_usd <- get_data("EXR.D.USD.EUR.SP00.A")   # USD per 1 EUR
  eur_gbp <- get_data("EXR.D.GBP.EUR.SP00.A")   # GBP per 1 EUR
  eur_usd_xts <- xts(as.numeric(eur_usd$obsvalue), order.by = as.Date(eur_usd$obstime))
  gbp_eur_xts <- xts(as.numeric(eur_gbp$obsvalue), order.by = as.Date(eur_gbp$obstime))
  getSymbols("DEXUSUK", src = "FRED")            # USD per 1 GBP
  usd_gbp_xts <- DEXUSUK
}



# Three series side by side (3-way merge = outer join, so calendar gaps are visible)
fx_raw <- merge(gbp_eur_xts, eur_usd_xts, usd_gbp_xts)
colnames(fx_raw) <- names(PAIR_COLS)
fx_raw <- fx_raw[paste0(TRAIN_START, "/", COVID_END)]

na_report <- data.frame(
  pair        = colnames(fx_raw),
  dates_total = nrow(fx_raw),
  missing     = colSums(is.na(fx_raw)),
  missing_pct = round(100 * colSums(is.na(fx_raw)) / nrow(fx_raw), 2)
)

print(na_report, row.names = FALSE)

# Keep only days on which all three rates are quoted, so every pair is forecast on
# exactly the same days. The gaps are holidays that differ between the ECB and the Fed.
fx <- na.omit(fx_raw)

# Daily log returns (%) in the evaluation period: used to explain the pair ranking
# Needed only when we make the final decision/report of which pairing is the most predictable
fx_eval <- fx[paste0(EVAL_START, "/", COVID_END)]
log_ret <- na.omit(diff(log(fx_eval)) * 100)





# =============================================================================
# 2. OVERVIEW FIGURE: ALL PAIRS INDEXED TO 100, COVID EVENTS MARKED
# =============================================================================

to_long <- function(x) {
  data.frame(date = index(x), coredata(x), check.names = FALSE) %>%
    pivot_longer(-date, names_to = "pair", values_to = "value") %>%
    mutate(pair = factor(pair, levels = names(PAIR_COLS)))
}
fx_index <- xts(sweep(coredata(fx), 2, coredata(fx)[1, ], "/") * 100, order.by = index(fx))

event_caption <- paste(strwrap(paste(
  paste0(covid_events$id, " = ", covid_events$label, " (", format(covid_events$date, "%d %b %Y"), ")"),
  collapse = ";  "), width = 140), collapse = "\n")

p0 <- ggplot(to_long(fx_index), aes(date, value, colour = pair)) +
  annotate("rect", xmin = TRAIN_START, xmax = EVAL_START, ymin = -Inf, ymax = Inf,
           fill = "grey92") +
  annotate("text", x = TRAIN_START + 15, y = Inf, label = "initial training (2019)",
           hjust = 0, vjust = 1.5, size = 3, colour = "grey40") +
  geom_hline(yintercept = 100, colour = "grey70") +
  geom_line(linewidth = 0.5) +
  geom_vline(data = covid_events, aes(xintercept = date, linetype = type),
             colour = "grey45", linewidth = 0.35) +
  # labels alternate between two heights so close events (2 and 3) do not overlap
  geom_text(data = covid_events, aes(x = date, y = Inf, label = id, vjust = ifelse(id %% 2 == 0, 1.4, 3)),
            hjust = -0.3, size = 3, colour = "grey25", inherit.aes = FALSE) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  scale_linetype_manual(values = c(COVID = "dashed", Other = "dotted"), name = "Event type") +
  labs(title = paste("All pairs indexed to 100 on", format(index(fx)[1], "%d %b %Y")),
       subtitle = "\"X per 1 Y\" up = X weaker; down = X stronger",
       x = NULL, y = "Index (start = 100)", caption = event_caption) +
  theme_paper
save_plot(p0, "figA0_indexed_with_events.png")


# =============================================================================
# 3. FORECASTING Help
# =============================================================================

forecast_one <- function(model, y_train, h) {
  y <- ts(y_train, frequency = SEAS_PERIOD)
  tryCatch(suppressWarnings({
    fc <- switch(model,
      naive        = naive(y, h = h, level = 95),
      ses          = ses(y, h = h, level = 95),
      holt         = holt(y, h = h, level = 95),
      holt_winters = hw(y, h = h, seasonal = "additive", level = 95),
      arima        = forecast(auto.arima(y, seasonal = FALSE), h = h, level = 95),
      sarima       = forecast(auto.arima(y, seasonal = TRUE),  h = h, level = 95))
    list(mean = as.numeric(fc$mean), lo = as.numeric(fc$lower[, 1]),
         hi = as.numeric(fc$upper[, 1]), method = fc$method)
  }), error = function(e) {
    list(mean = rep(NA_real_, h), lo = rep(NA_real_, h), hi = rep(NA_real_, h),
         method = paste("failed:", conditionMessage(e)))
  })
}

# ---- Rolling-origin CV, expanding window, for one series -------------------
# origins = position of the LAST training day of each fold. Training always starts
# at day 1, so the training window grows by CV_STEP days each fold.
rolling_cv <- function(y, dates, origins, h) {
  rows <- list()
  for (f in seq_along(origins)) {
    o   <- origins[f]
    h_f <- min(h, length(y) - o)            # the last fold may be shorter
    if (h_f < 1) next
    y_train  <- y[1:o]
    test_idx <- (o + 1):(o + h_f)
    scale    <- mean(abs(diff(y_train)))     # in-sample 1-step naive error, for MASE
    for (m in MODELS) {
      fc <- forecast_one(m, y_train, h_f)
      rows[[length(rows) + 1]] <- data.frame(
        fold = f, origin_date = dates[o], target_date = dates[test_idx],
        horizon = seq_len(h_f), model = m, method = fc$method,
        train_n = length(y_train), actual = y[test_idx], forecast = fc$mean,
        lo95 = fc$lo, hi95 = fc$hi, mase_scale = scale)
    }
  }
  do.call(rbind, rows)
}

run_cv_all_pairs <- function(x, origins, h) {
  out <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
    res <- rolling_cv(as.numeric(x[, p]), index(x), origins, h)
    res$pair <- p
    res
  }))
  out$pair <- factor(out$pair, levels = names(PAIR_COLS))
  out
}

# ---- Error metrics ----------------------------------------------------------
#   MAPE = mean absolute % error
#   MASE = MAE divided by the in-sample one-step naive error (Hyndman & Koehler 2006)
#   RMSE_vs_naive = RMSE of the model / RMSE of the random walk in the same group
#                   (< 1 means the model beats the random walk)
#   coverage95 = % of actual values inside the 95% prediction interval

summarise_errors <- function(df, ...) {
  df %>%
    group_by(...) %>%
    summarise(
      n_points   = n(),
      failed     = sum(is.na(forecast)),
      MSE        = mean((actual - forecast)^2, na.rm = TRUE),
      RMSE       = sqrt(MSE),
      MAE        = mean(abs(actual - forecast), na.rm = TRUE),
      MAPE       = 100 * mean(abs((actual - forecast) / actual), na.rm = TRUE),
      MASE       = mean(abs(actual - forecast) / mase_scale, na.rm = TRUE),
      coverage95 = 100 * mean(actual >= lo95 & actual <= hi95, na.rm = TRUE),
      .groups = "drop")
}
add_vs_naive <- function(s, keys) {
  base <- s %>% filter(model == "naive") %>% select(all_of(keys), RMSE_naive = RMSE)
  s %>% left_join(base, by = keys) %>%
    mutate(RMSE_vs_naive = RMSE / RMSE_naive) %>% select(-RMSE_naive)
}
order_models <- function(df) mutate(df, model = factor(model, levels = MODELS))


# =============================================================================
# 4. ANALYSIS A: THE WHOLE COVID PERIOD
# =============================================================================
# Fold 1 trains on 2019 and forecasts the first 20 trading days of 2020.
# Fold 2 trains on 2019 + those 20 days and forecasts the next 20, and so on
# until 5 May 2023.
first_origin <- max(which(index(fx) < EVAL_START))     # last trading day of 2019
origins_A    <- seq(first_origin, nrow(fx) - 1, by = CV_STEP)


t0 <- Sys.time()
cv_A <- run_cv_all_pairs(fx, origins_A, CV_H)
cat("  done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")

# Overall: one row per pair x model, all horizons pooled
A_overall <- summarise_errors(cv_A, pair, model) %>%
  add_vs_naive("pair") %>% order_models() %>% arrange(pair, model)

# By forecast horizon: 1 day, 1 week, 2 weeks, 1 month ahead
A_horizon <- summarise_errors(filter(cv_A, horizon %in% c(1, 5, 10, 20)), pair, model, horizon) %>%
  add_vs_naive(c("pair", "horizon")) %>% order_models() %>% arrange(pair, model, horizon)


print(A_overall %>% select(pair, model, RMSE, MAE, MAPE, MASE, RMSE_vs_naive, coverage95),
      n = Inf, width = Inf)



# Figure A1: every model relative to the random walk (below 1 = better)
pA1 <- ggplot(A_overall, aes(model, RMSE_vs_naive, colour = pair)) +
  geom_hline(yintercept = 1, colour = "grey50") +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = "COVID period: RMSE relative to the random walk",
       subtitle = "Below 1 = better than 'tomorrow equals today'",
       x = NULL, y = "RMSE / RMSE(naive)") +
  theme_paper
save_plot(pA1, "figA1_covid_vs_naive.png", h = 5)

# Figure A2: MAPE by horizon, one panel per model
pA2 <- ggplot(A_horizon, aes(horizon, MAPE, colour = pair)) +
  geom_line(linewidth = 0.6) + geom_point(size = 2) +
  facet_wrap(~ model, nrow = 2) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  scale_x_continuous(breaks = c(1, 5, 10, 20)) +
  labs(title = "COVID period: forecast error by horizon",
       x = "Forecast horizon (trading days)", y = "MAPE (%)") +
  theme_paper
save_plot(pA2, "figA2_covid_mape_by_horizon.png", h = 6)


# =============================================================================
# 5. WHICH PAIR IS MOST PREDICTABLE?
# =============================================================================
# MAPE is used because it is in %, so pairs with different price levels can be compared.
# Volatility is added because a pair that moves more is naturally harder to forecast:
# if the ranking by MAPE matches the ranking by volatility, the difference between
# pairs comes from how much they move, not from the models working better on one pair.
pair_summary <- A_overall %>%
  group_by(pair) %>%
  summarise(
    MAPE_naive        = MAPE[model == "naive"],
    MAPE_mean_models  = mean(MAPE),
    best_model        = as.character(model[which.min(MAPE)]),
    MAPE_best         = min(MAPE),
    best_RMSE_vs_naive = min(RMSE_vs_naive),
    .groups = "drop") %>%
  left_join(data.frame(pair = factor(names(PAIR_COLS), levels = names(PAIR_COLS)),
                       daily_vol_pct = apply(log_ret, 2, sd)[names(PAIR_COLS)]),
            by = "pair") %>%
  mutate(MAPE_per_vol = MAPE_naive / daily_vol_pct) %>%   # error per unit of volatility
  arrange(MAPE_mean_models)


print(pair_summary, width = Inf)

# Figure A3: MAPE per pair for every model
pA3 <- ggplot(A_overall, aes(pair, MAPE, colour = pair, shape = model)) +
  geom_point(size = 3, position = position_dodge(width = 0.6)) +
  scale_colour_manual(values = PAIR_COLS, guide = "none") +
  scale_x_discrete(labels = PAIR_LABELS) +
  labs(title = "COVID period: forecast error per currency pair",
       subtitle = "Each point is one model; MAPE pooled over horizons 1-20",
       x = NULL, y = "MAPE (%)", shape = "Model") +
  theme_paper
save_plot(pA3, "figA3_covid_mape_by_pair.png", h = 5)


# =============================================================================
# 6. FITTED PARAMETERS PER MODEL AND PAIR
# =============================================================================
# Each model fitted once on the full data (2019 to May 2023), for reporting.
# ETS: alpha (level), beta (trend), gamma (season). ARIMA: order and coefficients.
# AICc is only comparable within a family (ETS with ETS, ARIMA with ARIMA).
get_params <- function(fc) {
  m <- fc$model
  if (inherits(m, "ets")) {
    p <- m$par[intersect(c("alpha", "beta", "gamma"), names(m$par))]
  } else if (inherits(m, "Arima")) {
    p <- coef(m)
  } else {
    p <- numeric(0)                      # naive: nothing to estimate
  }
  data.frame(method = fc$method,
             params = if (length(p)) paste(names(p), round(p, 4), sep = " = ", collapse = ", ")
                      else "none",
             sigma  = if (!is.null(m$sigma2)) sqrt(m$sigma2) else NA,
             AICc   = if (!is.null(m$aicc)) m$aicc else NA)
}

param_table <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
  y <- ts(as.numeric(fx[, p]), frequency = SEAS_PERIOD)
  fits <- list(
    naive        = naive(y, h = 1),
    ses          = ses(y, h = 1),
    holt         = holt(y, h = 1),
    holt_winters = hw(y, h = 1, seasonal = "additive"),
    arima        = forecast(auto.arima(y, seasonal = FALSE), h = 1),
    sarima       = forecast(auto.arima(y, seasonal = TRUE),  h = 1))
  do.call(rbind, lapply(names(fits), function(m) data.frame(pair = p, model = m, get_params(fits[[m]]))))
}))

print(param_table, row.names = FALSE, digits = 4)


# =============================================================================
# 7. EXPORT
# =============================================================================
save_table(na_report,    "tab_covid_missing_values.csv")
save_table(A_overall,    "tab_covid_A_whole_period.csv")
save_table(A_horizon,    "tab_covid_A_by_horizon.csv")
save_table(pair_summary, "tab_covid_A_pair_ranking.csv")
save_table(param_table,  "tab_covid_fitted_parameters.csv")
save_table(cv_A %>% count(pair, model, method, name = "n_fits"), "tab_covid_model_choices.csv")
save_table(cv_A,         "raw_covid_forecasts_A.csv")

