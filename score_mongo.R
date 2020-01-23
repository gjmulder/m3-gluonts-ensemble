library(mongolite)
library(jsonlite)
library(matrixStats)
library(lubridate)
library(gtools)
library(dplyr)
library(ggplot2)

options(width = 160)

dataset_version <- Sys.getenv(c("DATASET", "VERSION"))
data_set <- as.character(dataset_version[1])
version <- as.character(dataset_version[2])

if (grepl("monthly", data_set)) {
  frq = 12
} else if (grepl("quarterly", data_set)) {
  frq = 4
} else if (grepl("yearly", data_set)) {
  frq = 1
} else if (grepl("other", data_set)) {
  frq = 1
} else {
  stop("Unrecognised data set. Can't determine frequency")
}
print(paste0("Determined frequency from data set: ", frq))

con <-
  mongo("jobs",
        paste0(data_set, "-", version),
        url = "mongodb://heika:27017",
        verbose = TRUE)
mydata <- con$find()
print(mydata$result$cfg$model$type)
rm(con)
gc()

model_types <- mydata$result$cfg$model$type
res <-
  readLines(paste0("/var/tmp/", data_set, "_all/test/data.json"))

smape_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <-
    as.numeric(outsample)
  forecasts <- as.numeric(forecasts)
  smape <-
    (abs(outsample - forecasts) * 200) / (abs(outsample) + abs(forecasts))
  return(smape)
}

mase_cal <- function(insample, outsample, forecasts, frq) {
  #Used to estimate MASE
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
  fcasts_df <- mydata$result$err_metrics$y_hats[col_idx,]
  smapes <- c()
  mases <- c()

  for (idx in c(1:length(fcasts_df))) {
    # print(idx)
    ts_fcast <-
      as.matrix(as.data.frame(Filter(Negate(is.null), fcasts_df[[idx]])))
    y_hat <- rowMedians(ts_fcast)

    test_row <- fromJSON(res[idx])
    in_sample <-
      head(test_row$target, length(test_row$target) - length(y_hat))
    y_test <- tail(test_row$target, length(y_hat))

    smapes <- c(smapes, smape_cal(y_test, y_hat))
    mases <- c(mases, mase_cal(in_sample, y_test, y_hat, frq))
  }

  return(list(sMAPE = mean(smapes),
              MASE = mean(mases)))
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
    print(comb[idx2,])
    col_idx <- c()
    for (model_type in comb[idx2,]) {
      model_col_idx <- which(model_type == model_types)
      col_idx <- c(col_idx, sample(model_col_idx, 50))
      print(length(col_idx))
    }
    errs <- ensemble_fcasts(col_idx)
    result <-
      data.frame(
        model.type = paste0(comb[idx2,], collapse = "+"),
        MASE    = errs[['MASE']],
        sMAPE   = errs[['sMAPE']]
      )
    print(result)
    results <- rbind(results, result)
  }
}
write.csv(results,
          file = "ensemble_combination_results.csv",
          row.names = FALSE,
          quote = FALSE)
print(results)

# Sampling
# col_idx_all <- which(!is.na(model_types) & model_types == "TransformerEstimator")
col_idx_all <-
  which(!is.na(model_types))
results <-
  data.frame(
    model.type = character(),
    number.models = integer(),
    MASE = double(),
    sMAPE = double()
  )
for (num_samples in c(10:length(col_idx_all))) {
  col_idx <- sample(col_idx_all, size = num_samples)
  errs <- ensemble_fcasts(col_idx)
  result <-
    data.frame(number.models = length(col_idx),
               MASE    = errs[['MASE']])
  print(result)
  results <- rbind(results, result)
}
write.csv(results,
          file = "ensemble_err_vs_size.csv",
          row.names = FALSE,
          quote = FALSE)
print(results)

gg <-
  ggplot(results, aes(x = number.models, y = MASE)) +
  geom_point(size = 0.1) +
  geom_smooth(size = 0.5, se = FALSE) +
  labs(title = "Forecast MASE versus number of ensembled models",
       x = "Number of models",
       y = "MASE")
print(gg)

tibble(
  model.type = model_types,
  book.time = ymd_hms(mydata$book_time, tz = "EET"),
  refresh.time = ymd_hms(mydata$refresh_time, tz = "EET")
) %>%
  mutate(duration = refresh.time - book.time) %>%
  group_by(model.type) %>%
  summarise(mean(duration)) ->
  timings

write.csv(timings,
          file = "timings.csv",
          row.names = FALSE,
          quote = FALSE)
print(timings)

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
