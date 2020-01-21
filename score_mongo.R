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
res <- readLines("/var/tmp/m3_monthly_all/test/data.json")

smape_cal <- function(outsample, forecasts){
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  smape <- (abs(outsample-forecasts)*200)/(abs(outsample)+abs(forecasts))
  return(smape)
}

mase_cal <- function(insample, outsample, forecasts){
  #Used to estimate MASE
  frq <- 12
  forecastsNaiveSD <- rep(NA,frq)
  for (j in (frq+1):length(insample)){
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j-frq])
  }
  masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)

  outsample <- as.numeric(outsample) ; forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample-forecasts))/masep
  return(mase)
}

ensemble_fcasts <- function(col_idx) {
  fcasts_df <- mydata$result$err_metrics$y_hats[col_idx,]
  smapes <- c()
  mases <- c()
  for (idx in c(1:length(fcasts_df))) {
    # print(idx)
    ts_fcast <-
      as.matrix(as.data.frame(Filter(Negate(is.null), fcasts_df[[idx]])))
    y_hat <- rowMedians(ts_fcast)

    test_row <- fromJSON(res[idx])
    in_sample <- head(test_row$target, length(test_row$target) - 18)
    y_hat_test <- tail(test_row$target, 18)

    smapes <- c(smapes, smape_cal(y_hat_test, y_hat))
    mases <- c(mases, mase_cal(in_sample, y_hat_test, y_hat))
  }
  return(list(sMAPE = mean(smapes), MASE = mean(mases)))
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
  comb <- combinations(length(uniq_model_types), idx1, uniq_model_types)
  for (idx2 in 1:nrow(comb)) {
    print(comb[idx2,])
    col_idx <- c()
    for (model_type in comb[idx2,]) {
      col_idx <- c(col_idx, which(model_type == model_types))
      print(length(col_idx))
    }
    errs <- ensemble_fcasts(col_idx)
    result <-
      data.frame(
        model.type = paste0(comb[idx2,], collapse = "+"),
        number.models = length(col_idx),
        MASE = errs[['MASE']],
        sMAPE = errs[['sMAPE']]
      )
    # print(result)
    results <- rbind(results, result)
  }
}
write.csv(results, file = "ensemble_results.csv", row.names = FALSE, quote = FALSE)

# # Ensembled, leave one model type out
# for (model_type in uniq_model_types) {
#   col_idx <- which(!is.na(model_types) & (model_type != model_types))
#   model_type <-
#     paste0(sort(unique(model_types[col_idx]), na.last = NA), collapse = "+")
#   errs <- ensemble_fcasts(col_idx)
#   result <-
#     data.frame(
#       model.type = model_type,
#       number.models = length(col_idx),
#       sMAPE = errs[['sMAPE']],
#       MASE = errs[['MASE']]
#     )
#   print(result)
#   results <- rbind(results, result)
# }
#
# # Ensembled DeepAREstimator + TransformerEstimator
# col_idx <-
#   which(model_types %in% c("DeepAREstimator", "TransformerEstimator"))
# model_type <-
#   paste0(c("DeepAREstimator", "TransformerEstimator"), collapse = "+")
# result <-
#   data.frame(
#     model.type = model_type,
#     number.models = length(col_idx),
#     sMAPE = errs[['sMAPE']],
#     MASE = errs[['MASE']]
#   )
# print(result)
# results <- rbind(results, result)
#
# # Ensembled DeepAREstimator + TransformerEstimator
# col_idx <-
#   which(model_types %in% c("DeepAREstimator", "TransformerEstimator"))
# model_type <-
#   paste0(c("DeepAREstimator", "TransformerEstimator"), collapse = "+")
# result <-
#   data.frame(
#     model.type = model_type,
#     number.models = length(col_idx),
#     sMAPE = errs[['sMAPE']],
#     MASE = errs[['MASE']]
#   )
# print(result)
# results <- rbind(results, result)
#
# # Ensembled DeepAREstimator + TransformerEstimator
# col_idx <-
#   which(model_types %in% c("DeepAREstimator", "TransformerEstimator"))
# model_type <-
#   paste0(c("DeepAREstimator", "TransformerEstimator"), collapse = "+")
# result <-
#   data.frame(
#     model.type = model_type,
#     number.models = length(col_idx),
#     sMAPE = errs[['sMAPE']],
#     MASE = errs[['MASE']]
#   )
# print(result)
# results <- rbind(results, result)
#
# # Ensembled everything
# col_idx <- which(!is.na(model_types))
# model_type <- paste0(uniq_model_types, collapse = "+")
# result <-
#   data.frame(
#     model.type = model_type,
#     number.models = length(col_idx),
#     sMAPE = errs[['sMAPE']],
#     MASE = errs[['MASE']]
#   )
# print(result)
# results <- rbind(results, result)

# # Sampling
# col_idx_all <- which(!is.na(model_types))
# for (num_samples in c(10:length(col_idx_all))) {
#   col_idx <- sample(col_idx_all, size = num_samples)
#   errs <- ensemble_fcasts(col_idx)
#   result <-
#     data.frame(
#       number.models = length(col_idx),
#       sMAPE = errs[['sMAPE']],
#       MASE = errs[['MASE']]
#     )
#   print(result)
#   results <- rbind(results, result)
# }
# print(results)

# gg <-
#   ggplot(results, aes(x = number.models, y = MASE)) +
#   geom_point(size = 0.1) +
#   geom_smooth(size = 0.5, se = FALSE) +
#   labs(
#     title = "18 month forecast MASE versus number of ensembled models",
#     x = "Number of models",
#     y = "18 month MASE"
#   )
# print(gg)

# tibble(model.type = model_types, book.time = ymd_hms(mydata$book_time, tz = "EET"), refresh.time = ymd_hms(mydata$refresh_time, tz = "EET")) %>%
#   mutate(duration = refresh.time - book.time) ->
#   timings
#
# timings %>%
#   group_by(model.type) %>%
#   summarise(mean(duration))
