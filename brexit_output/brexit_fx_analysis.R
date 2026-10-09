#  Brexit and exchange rates, 3 things we aim to answer:
#  (1) Is one currency pair more predictable than another?
#  (2) Do Holt / Holt-Winters/ ARIMA/ SARIMA forecast better than random walk?
#  (3) Does that change when we zoom in on single events and train on less data?

#  Structure:
#    0. Packages and settings (all tunable numbers are here)
#    1. Data: load or reuse your objects, align the three series, NA report
#    2. Descriptive plots with the Brexit events marked in every series,
#       plus an "event reaction" table (how far did each pair move?)
#    3. Box-Jenkins diagnostics per pair (ADF, chosen ARIMA, residual tests)
#    4. Forecasting involves: six models, error metrics, rolling-origin CV
#    5. Analysis A: the whole Brexit period (2015-2020)
#    6. Analysis B: the referendum +/- 3 months (before vs after)
#    7. Analysis C: 3-week windows around single events, trained on 20/60/120 days
#    8. Exports




#############################
# 0. PACKAGES AND SETTINGS
#############################
# install.packages(c("xts", "zoo", "forecast", "tseries", "ggplot2", "dplyr", "tidyr"))
library(xts)       # time-indexed series
library(zoo)       # rolling windows
library(forecast)  # naive(), ses(), holt(), hw(), auto.arima(), forecast()
library(tseries)   # adf.test()
library(ggplot2)
library(dplyr)
library(tidyr)
library(ecb)
library(quantmod)

# Periods
BREXIT_START <- as.Date("2015-01-01")
BREXIT_END   <- as.Date("2020-12-31")
DATA_END     <- as.Date("2021-02-26")  # extra weeks so event 8 (2020-12-31) has data after it
REFERENDUM   <- as.Date("2016-06-23")  # vote day: polls closed 22:00 UK time, so the 23 June quotes are still pre-result
REF_RESULT   <- as.Date("2016-06-24")  # first trading day that knows the result -> used to split before/after
REF_WINDOW   <- c(seq(REFERENDUM, by = "-3 months", length.out = 2)[2],   # 2016-03-23
                  seq(REFERENDUM, by =  "3 months", length.out = 2)[2])   # 2016-09-23
COVID_START  <- as.Date("2020-02-21")  # roughly when the COVID sell-off in markets starts

#### Brexit events 
events <- data.frame(
  id    = 1:8,
  date  = as.Date(c("2016-02-20", "2016-06-23", "2017-03-29", "2019-01-15",
                    "2019-10-17", "2019-12-12", "2020-01-31", "2020-12-31")),
  label = c("Referendum date announced", "Referendum day", "Article 50 triggered",
            "First meaningful vote rejected", "Revised deal agreed",
            "General election", "Formal exit", "Transition period ends"),
  # First trading day on which markets could react. Differs from the event date when the
  # event is on a weekend (20 Feb 2016 was a Saturday) or the result came overnight
  # (referendum, general election). Event reactions and event windows start from here.
  react = as.Date(c("2016-02-22", "2016-06-24", "2017-03-29", "2019-01-15",
                    "2019-10-17", "2019-12-13", "2020-01-31", "2020-12-31"))
)

##### Models 
# Daily currency data only has one possible "season": the 5-day trading week.
# Holt-Winters and SARIMA need a seasonal period, so we give them 5.
# If they do no better than Holt / ARIMA, that is a result: there is no weekly seasonality.
# As reported in the paper this does not mean there is no seasonality at all
SEAS_PERIOD <- 5
MODELS <- c("naive",         # random walk benchmark: tomorrow = today 
            "ses",           # simple exponential smoothing, ETS(A,N,N)
            "holt",          # Holt's linear trend, ETS(A,A,N)
            "holt_winters",  # Holt-Winters additive, ETS(A,A,A) with period 5
            "arima",         # auto.arima, non-seasonal
            "sarima")        # auto.arima, allowed to add seasonal terms with period 5

# ---- CV (expanding window, rolling origin)
# A: whole Brexit period. First origin after ~1 year (250 trading days),
#    then a new origin every 20 trading days, each forecasting 20 days (~1 month) ahead.
CV_INITIAL_A <- 250
CV_H_A       <- 20
CV_STEP_A    <- 20
# B: referendum window. 40 days initial training, forecast 1 week, move 1 week at a time.
CV_INITIAL_B <- 40
CV_H_B       <- 5
CV_STEP_B    <- 5
# C: events. 3 folds x 5 days = the 3 weeks starting at the event,
#    trained on the last 20 / 60 / 120 trading days before it (~1 / 3 / 6 months).
EVENT_LOOKBACKS <- c(20, 60, 120)
EVENT_H         <- 5
EVENT_FOLDS     <- 3

# ---- Output and plot style -------------------------------------------------
OUT_DIR <- "C:\\Users\\e_cic\\Desktop\\IS_seminar\\brexit_output"
dir.create(OUT_DIR, showWarnings = FALSE)

# One fixed colour per currency pair, used in every figure
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
# Create gbp_eur_xts, eur_usd_xts and usd_gbp_xts.

if (!all(sapply(c("gbp_eur_xts", "eur_usd_xts", "usd_gbp_xts"), exists))) {
  message("Downloading data from the ECB and FRED ...")
  eur_usd <- get_data("EXR.D.USD.EUR.SP00.A")   # USD per 1 EUR
  eur_gbp <- get_data("EXR.D.GBP.EUR.SP00.A")   # GBP per 1 EUR
  eur_usd_xts <- xts(as.numeric(eur_usd$obsvalue), order.by = as.Date(eur_usd$obstime))
  gbp_eur_xts <- xts(as.numeric(eur_gbp$obsvalue), order.by = as.Date(eur_gbp$obstime))
  getSymbols("DEXUSUK", src = "FRED")            # USD per 1 GBP
  usd_gbp_xts <- DEXUSUK
}


# Put the three series side by side on one date index.
# join = "outer" keeps every date where at least one series has a value,

fx_raw <- merge(gbp_eur_xts, eur_usd_xts, usd_gbp_xts)  
colnames(fx_raw) <- names(PAIR_COLS)
fx_raw <- fx_raw[paste0(BREXIT_START, "/", DATA_END)]

# ---- Missing value report --------------------------------------------------
na_report <- data.frame(
  pair         = colnames(fx_raw),
  quote        = PAIR_LABELS[colnames(fx_raw)],
  dates_total  = nrow(fx_raw),
  missing      = colSums(is.na(fx_raw)),
  missing_pct  = round(100 * colSums(is.na(fx_raw)) / nrow(fx_raw), 2)
)

print(na_report, row.names = FALSE)
for (p in colnames(fx_raw)) {
  miss <- index(fx_raw)[is.na(fx_raw[, p])]
  if (length(miss) > 0) cat("\n", p, "missing on:", format(head(miss, 15)),
                            if (length(miss) > 15) "...", "\n")
}

# Keep only days where ALL three are quoted, so every pair is modelled on exactly
# the same days (fair comparison). The gaps are bank holidays that differ between
# the ECB (TARGET calendar) and the Fed, not real data problems.

fx <- na.omit(fx_raw)


fx_brexit <- fx[paste0(BREXIT_START, "/", BREXIT_END)]
fx_ref    <- fx[paste0(REF_WINDOW[1], "/", REF_WINDOW[2])]

# Descriptive statistics per pair
desc_stats <- data.frame(
  pair = names(PAIR_COLS),
  n    = colSums(!is.na(fx_brexit)),
  mean = colMeans(fx_brexit),
  sd   = apply(fx_brexit, 2, sd),
  min  = apply(fx_brexit, 2, min),
  max  = apply(fx_brexit, 2, max)
)

print(desc_stats, row.names = FALSE, digits = 4)


# =============================================================================
# 2. DESCRIPTIVE PLOTS WITH EVENTS
# =============================================================================
# Helper: xts with several columns -> long data frame for ggplot
to_long <- function(x) {
  data.frame(date = index(x), coredata(x), check.names = FALSE) %>%
    pivot_longer(-date, names_to = "pair", values_to = "value") %>%
    mutate(pair = factor(pair, levels = names(PAIR_COLS)))
}

# Helper: dashed line + event number at the top of every panel.
# Numbers instead of text labels, because several events are close together
# (Oct/Dec 2019, Jan 2020) and text would overlap. The caption maps number -> event.
event_layers <- function(ev = events) {
  list(
    geom_vline(data = ev, aes(xintercept = date),
               linetype = "dashed", colour = "grey45", linewidth = 0.35),
    geom_text(data = ev, aes(x = date, y = Inf, label = id),
              vjust = 1.4, hjust = -0.3, size = 3, colour = "grey25")
  )
}
event_caption <- paste(strwrap(paste(
  paste0(events$id, " = ", events$label, " (", format(events$date, "%d %b %Y"), ")"),
  collapse = ";  "), width = 140), collapse = "\n")

# ---- Figure 1: levels of all three pairs, events marked in every panel -----
p1 <- ggplot(to_long(fx_brexit), aes(date, value, colour = pair)) +
  geom_line(linewidth = 0.4) +
  event_layers() +
  facet_wrap(~ pair, ncol = 1, scales = "free_y", labeller = as_labeller(PAIR_LABELS)) +
  scale_colour_manual(values = PAIR_COLS, guide = "none") +
  labs(title = "Exchange rates during Brexit, 2015-2020",
       x = NULL, y = "Exchange rate", caption = event_caption) +
  theme_paper
save_plot(p1, "fig1_levels_with_events.png", h = 8)

# ---- Figure 2: all pairs indexed to 100 on the first day -------------------
# Different price levels cannot share one axis, but indexed series can:
# the y-axis is "% of the starting value", so the size of the moves is comparable.

fx_index <- xts(sweep(coredata(fx_brexit), 2, coredata(fx_brexit)[1, ], "/") * 100,
                order.by = index(fx_brexit))
p2 <- ggplot(to_long(fx_index), aes(date, value, colour = pair)) +
  geom_hline(yintercept = 100, colour = "grey70") +
  geom_line(linewidth = 0.5) +
  event_layers() +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = paste("Which currency moved most? All pairs indexed to 100 on", format(index(fx_brexit)[1], "%d %b %Y")),
       subtitle = "GBP per EUR up = pound weaker vs euro; USD per GBP down = pound weaker vs dollar",
       x = NULL, y = "Index (start = 100)", caption = event_caption) +
  theme_paper
save_plot(p2, "fig2_indexed_with_events.png")

#  Figure 3: rolling volatility (how nervous was the market?)
# Daily log returns in %, then a 20-day rolling standard deviation.
log_ret <- na.omit(diff(log(fx_brexit)) * 100)
vol_20  <- na.omit(rollapply(log_ret, width = 20, FUN = sd, align = "right", by.column = TRUE))
p3 <- ggplot(to_long(vol_20), aes(date, value, colour = pair)) +
  geom_line(linewidth = 0.45) +
  event_layers() +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = "20-day rolling volatility of daily log returns",
       x = NULL, y = "Std. dev. of daily return (%)", caption = event_caption) +
  theme_paper
save_plot(p3, "fig3_volatility_with_events.png")

#  Figure 4: zoom on the referendum, +- 3 months
4 <- ggplot(to_long(fx_ref), aes(date, value, colour = pair)) +
  geom_line(linewidth = 0.5) +
  geom_vline(xintercept = REFERENDUM, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ pair, ncol = 1, scales = "free_y", labeller = as_labeller(PAIR_LABELS)) +
  scale_colour_manual(values = PAIR_COLS, guide = "none") +
  labs(title = "Three months before and after the referendum (dashed line = 23 June 2016)",
       x = NULL, y = "Exchange rate") +
  theme_paper
save_plot(p4, "fig4_referendum_window.png", h = 7)

#  Table: event reactions
# Move from the last quote before the market could react (events$react) to 1, 5 and
# 15 trading days later (log %),
# and the 1-day move expressed in "normal days" (divided by the daily return std. dev.).
sd_daily <- apply(log_ret, 2, sd)
event_reaction <- do.call(rbind, lapply(seq_len(nrow(events)), function(i) {
  pos0 <- max(which(index(fx) < events$react[i]))
  do.call(rbind, lapply(names(PAIR_COLS), function(p) {
    y  <- as.numeric(fx[, p])
    ch <- sapply(c(1, 5, 15), function(k)
      if (pos0 + k <= length(y)) 100 * log(y[pos0 + k] / y[pos0]) else NA)
    data.frame(event_id = i, event = events$label[i], react_from = events$react[i], pair = p,
               chg_1d_pct = ch[1], chg_1w_pct = ch[2], chg_3w_pct = ch[3],
               move_1d_in_sd = ch[1] / sd_daily[p])
  }))
}))

print(event_reaction %>% mutate(across(where(is.numeric) & !event_id, ~ round(.x, 2))),
      row.names = FALSE)
save_table(event_reaction, "tab_event_reactions.csv")


# =============================================================================
# 3. BOX-JENKINS DIAGNOSTICS PER PAIR (whole Brexit period)
# =============================================================================
# For each pair: is it stationary (ADF on levels and on differences)?
# Which ARIMA does auto.arima choose? Are the residuals white noise (Ljung-Box),
# normal (Shapiro-Wilk), and is there volatility clustering (Ljung-Box on squared residuals)?
diag_table <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
  y   <- as.numeric(fx_brexit[, p])
  fit <- auto.arima(ts(y, frequency = SEAS_PERIOD), seasonal = FALSE)
  k   <- sum(arimaorder(fit)[c("p", "q")])
  res <- residuals(fit)
  # Residual plots (ACF, histogram) for the appendix
  png(file.path(OUT_DIR, paste0("fig_resid_", p, ".png")), width = 1800, height = 1200, res = 200)
  checkresiduals(fit, test = FALSE)
  dev.off()
  data.frame(
    pair          = p,
    adf_p_levels  = suppressWarnings(adf.test(y)$p.value),        # large p: non-stationary
    adf_p_diff    = suppressWarnings(adf.test(diff(y))$p.value),  # small p: differences are stationary
    arima_chosen  = as.character(fit),
    AICc          = fit$aicc,
    ljung_box_p   = Box.test(res, lag = 10, type = "Ljung-Box", fitdf = k)$p.value,
    ljung_box_sq_p = Box.test(res^2, lag = 10, type = "Ljung-Box")$p.value,  # small p: ARCH effects
    shapiro_p     = shapiro.test(as.numeric(res))$p.value
  )
}))

print(diag_table, row.names = FALSE, digits = 3)
save_table(diag_table, "tab_diagnostics_whole_period.csv")

# =============================================================================
# 3b. FITTED PARAMETERS PER MODEL AND PAIR (whole Brexit period)
# =============================================================================
# Every model fitted once on the full period, so the parameters can be reported.
# ETS models: alpha (level), beta (trend), gamma (season). ARIMA: chosen order + coefficients.
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
             sigma  = if (!is.null(m$sigma2)) sqrt(m$sigma2) else NA,   # residual std. dev.
             AICc   = if (!is.null(m$aicc)) m$aicc else NA)
}

param_table <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
  y <- ts(as.numeric(fx_brexit[, p]), frequency = SEAS_PERIOD)
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
save_table(param_table, "tab_fitted_parameters.csv")


# =============================================================================
# 4. FORECASTING Help
# =============================================================================
# ---- One model, one training set, h steps ahead ----------------------------
# Returns point forecasts, 95% interval and a description of the fitted model.
# If a model cannot be fitted (e.g. Holt-Winters on 20 observations) it returns NA
# instead of crashing, and the failure is counted in the results.
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

# ---- Rolling-origin CV with an expanding window for one series -------------
# origins     = positions of the LAST training observation of each fold
# train_start = first training observation (fixed, so the window expands)
# The explanation is in the paper in more detail for a specific sereis,
# the idea is that we train on x amount of data points predict y, then train 
# on x+y (time & data points wise) and
# predict y+y (time wise but only y number of data points wise)
# Every forecast point is stored as one row, so any metric can be computed later
# for any grouping (per pair, model, horizon, era, event, ...).
# this is better explained in the paper 
rolling_cv <- function(y, dates, origins, h, train_start = 1) {
  rows <- list()
  for (f in seq_along(origins)) {
    o   <- origins[f]
    h_f <- min(h, length(y) - o)            # last fold may be shorter
    if (h_f < 1) next
    y_train  <- y[train_start:o]
    test_idx <- (o + 1):(o + h_f)
    scale    <- mean(abs(diff(y_train)))     # in-sample naive error, for MASE
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

# Same CV for all three pairs, with extra tag columns 
run_cv_all_pairs <- function(x, origins, h, train_start = 1, tags = list()) {
  out <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
    res <- rolling_cv(as.numeric(x[, p]), index(x), origins, h, train_start)
    res$pair <- p
    res
  }))
  out$pair <- factor(out$pair, levels = names(PAIR_COLS))   # same pair order everywhere
  for (nm in names(tags)) out[[nm]] <- tags[[nm]]
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

# Consistent model order in tables and plots
order_models <- function(df) mutate(df, model = factor(model, levels = MODELS))


# =============================================================================
# 5. ANALYSIS A: THE WHOLE BREXIT PERIOD
# =============================================================================
# Question: over 2015-2020 as a whole, which pair is most predictable and which model wins?
origins_A <- seq(CV_INITIAL_A, nrow(fx_brexit) - 1, by = CV_STEP_A)

t0 <- Sys.time()
cv_A <- run_cv_all_pairs(fx_brexit, origins_A, CV_H_A, tags = list(analysis = "A_whole"))

# Overall: pair x model
A_overall <- summarise_errors(cv_A, pair, model) %>%
  add_vs_naive("pair") %>% order_models() %>% arrange(pair, model)
# By forecast horizon (1 day ahead, 1 week, 1 month)
A_horizon <- summarise_errors(filter(cv_A, horizon %in% c(1, 5, 10, 20)), pair, model, horizon) %>%
  add_vs_naive(c("pair", "horizon")) %>% order_models()


print(A_overall %>% select(pair, model, RMSE, MAE, MAPE, MASE, RMSE_vs_naive, coverage95),
      n = Inf, width = Inf)

# Figure A1: model vs random walk, per pair (below 1 = beats the random walk)
pA1 <- ggplot(A_overall, aes(model, RMSE_vs_naive, colour = pair)) +
  geom_hline(yintercept = 1, colour = "grey50") +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = "Whole Brexit period: RMSE relative to the random walk",
       subtitle = "Below 1 = better than 'tomorrow equals today'",
       x = NULL, y = "RMSE / RMSE(naive)") +
  theme_paper
save_plot(pA1, "figA1_whole_period_vs_naive.png", h = 5)

# Figure A2: MAPE by horizon (scale-free, so pairs can be compared directly)
pA2 <- ggplot(A_horizon, aes(horizon, MAPE, colour = pair)) +
  geom_line(linewidth = 0.6) + geom_point(size = 2) +
  facet_wrap(~ model, nrow = 2) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  scale_x_continuous(breaks = c(1, 5, 10, 20)) +
  labs(title = "Whole Brexit period: forecast error grows with the horizon",
       x = "Forecast horizon (trading days)", y = "MAPE (%)") +
  theme_paper
save_plot(pA2, "figA2_mape_by_horizon.png", h = 6)


# =============================================================================
# 6. ANALYSIS B: THE REFERENDUM, +/- 3 MONTHS
# =============================================================================
pre <- index(fx_ref) < REF_RESULT   # "before" = everything up to and including 23 June
n_pre <- sum(pre); n_post <- sum(!pre)

# ---- B1: the shock test ----------------------------------------------------
# Train on the 3 months BEFORE the result only, then forecast the whole 3 months
# after it. This is what a forecaster on the evening of 23 June 2016 would have predicted.
cv_B1 <- run_cv_all_pairs(fx_ref, origins = n_pre, h = n_post, tags = list(analysis = "B1_shock"))
cv_B1$horizon_group <- cut(cv_B1$horizon, breaks = c(0, 1, 5, 20, Inf),
                           labels = c("day 1", "days 2-5", "days 6-20", "days 21+"))
B1_table <- summarise_errors(cv_B1, pair, model, horizon_group) %>%
  add_vs_naive(c("pair", "horizon_group")) %>% order_models()

print(B1_table %>% select(pair, model, horizon_group, RMSE, MAPE, MASE, coverage95),
      n = Inf, width = Inf)

# Figure B1: forecasts vs what actually happened, with 95% intervals
ref_actual <- to_long(fx_ref)
pB1 <- ggplot() +
  geom_ribbon(data = cv_B1 %>% filter(model != "naive", model != "ses") %>% order_models(),
              aes(target_date, ymin = lo95, ymax = hi95, fill = pair), alpha = 0.18) +
  geom_line(data = ref_actual, aes(date, value), colour = "grey20", linewidth = 0.4) +
  geom_line(data = cv_B1 %>% filter(model != "naive", model != "ses") %>% order_models(),
            aes(target_date, forecast, colour = pair), linewidth = 0.7) +
  geom_vline(xintercept = REFERENDUM, linetype = "dashed", colour = "grey40") +
  facet_grid(pair ~ model, scales = "free_y", labeller = labeller(pair = PAIR_LABELS)) +
  scale_colour_manual(values = PAIR_COLS, guide = "none") +
  scale_fill_manual(values = PAIR_COLS, guide = "none") +
  labs(title = "Forecasting the referendum aftermath with pre-referendum data only",
       subtitle = "Black = actual; coloured line = forecast; band = 95% prediction interval",
       x = NULL, y = "Exchange rate") +
  theme_paper + theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(pB1, "figB1_referendum_shock_forecasts.png", w = 12, h = 8)

# ---- B2: did the process itself change? ------------------------------------
# Fit the models separately on the 3 months before and the 3 months after.
# If the chosen ARIMA, the residual variance or the Holt smoothing parameters change,
# that is evidence of a structural break around the referendum.
B2_table <- do.call(rbind, lapply(names(PAIR_COLS), function(p) {
  do.call(rbind, lapply(c("before", "after"), function(ph) {
    y    <- as.numeric(fx_ref[if (ph == "before") pre else !pre, p])
    fitA <- auto.arima(ts(y, frequency = SEAS_PERIOD), seasonal = FALSE)
    fitH <- holt(ts(y, frequency = SEAS_PERIOD))$model
    k    <- sum(arimaorder(fitA)[c("p", "q")])
    data.frame(pair = p, phase = ph, n = length(y),
               mean_level     = mean(y),
               sd_daily_ret   = sd(diff(log(y))) * 100,   # volatility, % per day
               arima_chosen   = as.character(fitA),
               arima_sigma    = sqrt(fitA$sigma2),
               ljung_box_p    = Box.test(residuals(fitA), lag = 10, type = "Ljung-Box",
                                         fitdf = k)$p.value,
               holt_alpha     = unname(fitH$par["alpha"]),
               holt_beta      = unname(fitH$par["beta"]))
  }))
}))

print(B2_table, row.names = FALSE, digits = 3)

# ---- B3: rolling CV through the referendum window --------------------------
# 1-week-ahead forecasts, origin moved 1 week at a time. Each fold is labelled:
#   before      = the whole forecast week is before the referendum
#   contains    = the model has NOT seen the referendum yet but has to forecast through it
#   after       = the referendum is already in the training data
origins_B3 <- seq(CV_INITIAL_B, nrow(fx_ref) - 1, by = CV_STEP_B)
cv_B3 <- run_cv_all_pairs(fx_ref, origins_B3, CV_H_B, tags = list(analysis = "B3_rolling")) %>%
  group_by(pair, model, fold) %>%
  mutate(phase = case_when(max(target_date) < REF_RESULT ~ "1 before",
                           origin_date[1] >= REF_RESULT  ~ "3 after",
                           TRUE                          ~ "2 contains referendum")) %>%
  ungroup()
B3_table <- summarise_errors(cv_B3, phase, pair, model) %>%
  add_vs_naive(c("phase", "pair")) %>% order_models()

print(B3_table %>% select(phase, pair, model, RMSE, MAPE, MASE, RMSE_vs_naive, coverage95),
      n = Inf, width = Inf)

pB3 <- ggplot(B3_table, aes(model, MAPE, colour = pair)) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  facet_wrap(~ phase, nrow = 1, scales = "free_y") +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = "1-week-ahead error before, through and after the referendum",
       subtitle = "Note the free y-axes: the middle panel is on a much larger scale",
       x = NULL, y = "MAPE (%)") +
  theme_paper + theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(pB3, "figB3_rolling_by_phase.png", w = 11, h = 5)


# =============================================================================
# 7. ANALYSIS C: SHORT EVENT WINDOWS, DIFFERENT AMOUNTS OF TRAINING DATA
# =============================================================================
# For each event: the origin is the last quote before the market could react
# (events$react, e.g. 23 June 2016 for the referendum). We forecast
# the next 3 weeks as 3 folds of 5 days (expanding window), and repeat this with
# 20, 60 and 120 trading days of training data.
# Questions: which pair/model handles single events best, and does less data hurt?


cv_C <- do.call(rbind, lapply(seq_len(nrow(events)), function(i) {
  o0 <- max(which(index(fx) < events$react[i]))
  do.call(rbind, lapply(EVENT_LOOKBACKS, function(L) {
    origins <- o0 + (0:(EVENT_FOLDS - 1)) * EVENT_H
    run_cv_all_pairs(fx, origins, EVENT_H, train_start = max(1, o0 - L + 1),
                     tags = list(analysis = "C_events", event_id = i,
                                 event = events$label[i], lookback = L))
  }))
}))

# Pooled over all events: effect of the amount of training data
C_lookback <- summarise_errors(cv_C, lookback, pair, model) %>%
  add_vs_naive(c("lookback", "pair")) %>% order_models()
# Per event (every lookback)
C_event <- summarise_errors(cv_C, event_id, event, lookback, pair, model) %>%
  add_vs_naive(c("event_id", "event", "lookback", "pair")) %>% order_models()


print(C_lookback %>% select(lookback, pair, model, RMSE, MAPE, MASE, RMSE_vs_naive, failed),
      n = Inf, width = Inf)

# Figure C1: does more training data help?
pC1 <- ggplot(C_lookback, aes(factor(lookback), MASE, colour = pair, group = pair)) +
  geom_line(linewidth = 0.6) + geom_point(size = 2) +
  facet_wrap(~ model, nrow = 2) +
  scale_colour_manual(values = PAIR_COLS, labels = PAIR_LABELS, name = NULL) +
  labs(title = "Event windows: forecast error vs amount of training data",
       x = "Training days before the event", y = "MASE (pooled over 8 events)") +
  theme_paper
save_plot(pC1, "figC1_events_by_lookback.png", h = 6)

# Figure C2: heatmap, which model beats the random walk at which event (60-day training)
pC2 <- C_event %>% filter(lookback == 60) %>%
  mutate(event = factor(paste0(event_id, " ", event), levels = rev(paste0(events$id, " ", events$label)))) %>%
  ggplot(aes(model, event, fill = RMSE_vs_naive)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", RMSE_vs_naive)), size = 3) +
  facet_wrap(~ pair, labeller = as_labeller(PAIR_LABELS)) +
  scale_fill_gradient2(low = "#2166AC", mid = "grey96", high = "#B2182B", midpoint = 1,
                       name = "RMSE / naive") +
  labs(title = "Event windows (60 training days): RMSE relative to the random walk",
       subtitle = "Blue = better than the random walk, red = worse", x = NULL, y = NULL) +
  theme_paper + theme(axis.text.x = element_text(angle = 45, hjust = 1),
                      panel.grid = element_blank())
save_plot(pC2, "figC2_events_heatmap.png", w = 12, h = 6)


# =============================================================================
# 8. EXPORT AND SUMMARY
# =============================================================================
save_table(A_overall,  "tab_A_whole_period.csv")
save_table(A_horizon,  "tab_A_by_horizon.csv")
save_table(B1_table,   "tab_B1_referendum_shock.csv")
save_table(B2_table,   "tab_B2_before_vs_after.csv")
save_table(B3_table,   "tab_B3_rolling_by_phase.csv")
save_table(C_lookback, "tab_C_by_lookback.csv")
save_table(C_event,    "tab_C_by_event.csv")

# Which models did auto.arima / ets actually pick? Useful to report, e.g. how often
# SARIMA really used a seasonal term, or whether ARIMA just collapsed to a random walk.
model_choices <- bind_rows(cv_A, cv_B1, cv_B3, cv_C) %>%
  distinct(analysis, pair, model, fold, origin_date, method, event_id, lookback) %>%
  count(analysis, pair, model, method, name = "n_fits") %>%
  arrange(analysis, pair, model, desc(n_fits))
save_table(model_choices, "tab_model_choices.csv")

# Raw forecast errors, in case you want other groupings later
save_table(bind_rows(cv_A, cv_B1, cv_B3, cv_C), "raw_forecasts_all_analyses.csv")

# Best model per pair in each analysis (lowest MASE)
best <- bind_rows(
  A_overall  %>% mutate(analysis = "A whole period"),
  B1_table   %>% filter(horizon_group == "days 2-5") %>% mutate(analysis = "B1 shock, days 2-5"),
  B3_table   %>% mutate(analysis = paste("B3", phase)) %>% select(-phase),
  C_lookback %>% mutate(analysis = paste("C events, lookback", lookback)) %>% select(-lookback)
) %>%
  group_by(analysis, pair) %>% slice_min(MASE, n = 1, with_ties = FALSE) %>% ungroup() %>%
  select(analysis, pair, best_model = model, MASE, MAPE, RMSE_vs_naive)

print(best, n = Inf, width = Inf)
save_table(best, "tab_best_models.csv")

# Most predictable pair per analysis (average MASE over all models)
pair_rank <- bind_rows(
  A_overall  %>% mutate(analysis = "A whole period"),
  B3_table   %>% mutate(analysis = paste("B3", phase)),
  C_lookback %>% mutate(analysis = paste("C events, lookback", lookback))
) %>%
  group_by(analysis, pair) %>% summarise(mean_MASE = mean(MASE), mean_MAPE = mean(MAPE), .groups = "drop") %>%
  arrange(analysis, mean_MASE)

print(pair_rank, n = Inf, width = Inf)
save_table(pair_rank, "tab_pair_ranking.csv")

