library(mongolite)
library(jsonlite)
library(matrixStats)
# library(Metrics)
library(lubridate)
library(gtools)
library(dplyr)
library(ggplot2)

options(width = 150)

con <-
  mongo("jobs",
        "m3_monthly-final_rnd_6-18_mon",
        # "m3_monthly-final_rnd_1-step",
        url = "mongodb://heika:27017",
        verbose = TRUE)
mydata <- con$find()
print(mydata$result$cfg$model$type)
# fcasts <-
#   t(as.data.frame(Filter(
#     Negate(is.null), mydata$result$err_metrics$y_hats[[1]]
# )))
rm(con)
gc()

model_types <- mydata$result$cfg$model$type
res <- readLines("./m3_monthly_all/test/data.json")

smape_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <-
    as.numeric(outsample)
  forecasts <- as.numeric(forecasts)
  smape <-
    (abs(outsample - forecasts) * 200) / (abs(outsample) + abs(forecasts))
  return(smape)
}

mase_cal <- function(insample, outsample, forecasts) {
  #Used to estimate MASE
  frq <- 12
  forecastsNaiveSD <- rep(NA, frq)
  for (j in (frq + 1):length(insample)) {
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j - frq])
  }
  masep <- mean(abs(insample - forecastsNaiveSD), na.rm = TRUE)

  outsample <-
    as.numeric(outsample)
  forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample - forecasts)) / masep
  return(mase)
}

ensemble_fcasts <- function(col_idx) {
  fcasts_df <- mydata$result$err_metrics$y_hats[col_idx, ]
  smapes <- c()
  mases <- c()
  smapes6 <- c()
  mases6 <- c()
  smapes12 <- c()
  mases12 <- c()
  smapes18 <- c()
  mases18 <- c()

  for (idx in c(1:length(fcasts_df))) {
    # print(idx)
    ts_fcast <-
      as.matrix(as.data.frame(Filter(Negate(is.null), fcasts_df[[idx]])))
    y_hat <- rowMedians(ts_fcast)

    test_row <- fromJSON(res[idx])
    in_sample <- head(test_row$target, length(test_row$target) - 18)
    y_test <- tail(test_row$target, 18)

    smapes <- c(smapes, smape_cal(y_test, y_hat))
    mases <- c(mases, mase_cal(in_sample, y_test, y_hat))

    y_hat6  <- y_hat[1:6]
    y_test6 <- y_test[1:6]
    smapes6 <- c(smapes6, smape_cal(y_test6, y_hat6))
    mases6 <- c(mases6, mase_cal(in_sample, y_test6, y_hat6))

    y_hat12  <- y_hat[7:12]
    y_test12 <- y_test[7:12]
    smapes12 <- c(smapes12, smape_cal(y_test12, y_hat12))
    mases12 <- c(mases12, mase_cal(in_sample, y_test12, y_hat12))

    y_hat18  <- y_hat[13:18]
    y_test18 <- y_test[13:18]
    smapes18 <- c(smapes18, smape_cal(y_test18, y_hat18))
    mases18 <- c(mases18, mase_cal(in_sample, y_test18, y_hat18))

  }
  return(
    list(
      sMAPE = mean(smapes),
      MASE = mean(mases),
      sMAPE6 = mean(smapes6),
      MASE6 = mean(mases6),
      sMAPE12 = mean(smapes12),
      MASE12 = mean(mases12),
      sMAPE18 = mean(smapes18),
      MASE18 = mean(mases18)
    )
  )
}

uniq_model_types <- sort(unique(model_types))

results <-
  data.frame(
    model.type = character(),
    number.models = integer(),
    MASE = double(),
    sMAPE = double()
   )
for (idx1 in 1:length(uniq_model_types)) {
  comb <-
    combinations(length(uniq_model_types), idx1, uniq_model_types)
  for (idx2 in 1:nrow(comb)) {
    print(comb[idx2, ])
    col_idx <- c()
    for (model_type in comb[idx2, ]) {
      model_col_idx <- which(model_type == model_types)
      col_idx <- c(col_idx, sample(model_col_idx, 50))
      print(length(col_idx))
    }
    errs <- ensemble_fcasts(col_idx)
     result <-
      data.frame(
        model.type = paste0(comb[idx2, ], collapse = "+"),
        MASE6   = errs[['MASE6']],
        MASE12  = errs[['MASE12']],
        MASE18  = errs[['MASE18']],
        MASE    = errs[['MASE']],
        sMAPE6  = errs[['sMAPE6']],
        sMAPE12 = errs[['sMAPE12']],
        sMAPE18 = errs[['sMAPE18']],
        sMAPE   = errs[['sMAPE']]
      )
    print(result)
    results <- rbind(results, result)
  }
}
write.csv(results, file = "ensemble_results.csv", row.names = FALSE, quote = FALSE)
print(results)

## Sampling
## col_idx_all <- which(!is.na(model_types) & model_types == "TransformerEstimator")
#col_idx_all <-
#  which(!is.na(model_types))
#results <-
#  data.frame(
#    model.type = character(),
#    number.models = integer(),
#    MASE = double(),
#    sMAPE = double()
#  )
#for (num_samples in c(10:length(col_idx_all))) {
#  col_idx <- sample(col_idx_all, size = num_samples)
#  errs <- ensemble_fcasts(col_idx)
#  result <-
#    data.frame(
#      number.models = length(col_idx),
#      MASE    = errs[['MASE']]
#    )
#  print(result)
#  results <- rbind(results, result)
#}
# write.csv(results,
#           file = "ensemble_results.csv",
#           row.names = FALSE,
#           quote = FALSE)
# print(results)

gg <-
  ggplot(results, aes(x = number.models, y = MASE)) +
  geom_point(size = 0.1) +
  geom_smooth(size = 0.5, se = FALSE) +
  labs(title = "18 month forecast MASE versus number of ensembled models",
       x = "Number of models",
       y = "MASE")
print(gg)

# for (model_type in uniq_model_types) {
#   col_idx <- which(model_types == model_type)
#   MASEs <- tibble(MASE = mydata$result$err_metrics$validate$MASE[col_idx])
#   gg <-
#     ggplot(MASEs, aes(x = MASE)) +
#     geom_histogram(bins = 20) +
#     labs(title = paste0("Distribution of ", model_type, " MASEs"),
#          x = "MASE",
#          y = "Count")
#   print(gg)
# }

# tibble(model.type = model_types, book.time = ymd_hms(mydata$book_time, tz = "EET"), refresh.time = ymd_hms(mydata$refresh_time, tz = "EET")) %>%
#   mutate(duration = refresh.time - book.time) ->
#   timings
#
# timings %>%
#   group_by(model.type) %>%
#   summarise(mean(duration))
