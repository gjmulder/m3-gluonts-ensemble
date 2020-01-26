library(mongolite)
library(jsonlite)
library(matrixStats)
library(lubridate)
library(gtools)
library(dplyr)
library(ggplot2)

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
  # print(col_idx)
  fcasts_df <- mydata$result$err_metrics$y_hats[col_idx, ]
  smapes <- c()
  mases <- c()

  for (idx in c(1:length(fcasts_df))) {
    # print(idx)
    ts_fcast <-
      as.matrix(as.data.frame(Filter(Negate(is.null), fcasts_df[[idx]])))
    if (ncol(ts_fcast) == 0) {
      break
    }
    y_hat <- rowMedians(ts_fcast, na.rm = TRUE)

    test_row <- fromJSON(res[idx])
    in_sample <-
      head(test_row$target, length(test_row$target) - length(y_hat))
    y_test <- tail(test_row$target, length(y_hat))

    smapes <- c(smapes, mean(smape_cal(y_test, y_hat)))
    mases <- c(mases, mean(mase_cal(in_sample, y_test, y_hat, frq)))
  }
  return(list(sMAPE = mean(smapes),
              MASE = mean(mases)))
}

build_ensemble <- function(model_set) {
  uniq_model_types <- sort(unique(model_set$type))
  # print(uniq_model_types)

  results_comb <-
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
      col_idx <- c()
      for (model_type in comb[idx2, ]) {
        model_col_idx <- which(model_type == model_set$type)
        # col_idx <- c(col_idx, sample(model_col_idx, 50))
        col_idx <- c(col_idx, model_col_idx)
      }
      errs <- ensemble_fcasts(col_idx)
      result <-
        data.frame(
          model.type = paste0(comb[idx2, ], collapse = "+"),
          number.models = length(col_idx),
          MASE    = errs[['MASE']],
          sMAPE   = errs[['sMAPE']]
        )
      results_comb <- rbind(results_comb, result)
    }
  }
  return(results_comb)
}

if (!interactive()) {
  options(width = 1000)
}

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
rm(con)
gc()
type_loss <-
  tibble(type = mydata$result$cfg$model$type,
         loss = mydata$result$loss)

res <-
  readLines(paste0("/var/tmp/", data_set, "_all/test/data.json"))
  # readLines("/home/mulderg/Work/plos1-m3/m3_monthly_all_test/test/data.json")

for (n in c(1:8)*10) {
  # for (n in c(70, 80, 90, 100)) {
  type_loss %>%
    mutate(row.id = row_number()) %>%
    # filter(type == "DeepAREstimator") %>%
    # na.omit %>%
    group_by(type) %>%
    # sample_n(n, replace = FALSE) ->
    top_n(-n, loss) ->
    # top_frac(-n/100, loss) ->
    best_models
  # print(best_models)

  build_ensemble(best_models) %>%
    na.omit %>%
    mutate(model.type = gsub("Estimator", "", model.type)) %>%
    arrange(desc(sMAPE)) ->
    results_comb
  results_comb$n.star.models <- rep(n, nrow(results_comb))
  print(tail(results_comb, 5))
  # print(results_comb)
}
# model_types <- mydata$result$cfg$model$type

# write.csv(results_comb,
#           file = paste0(data_set, "-", version, "-ensemble_combination_results.csv"),
#           row.names = FALSE,
#           quote = FALSE)

# Sampling
# col_idx_all <- which(!is.na(model_types) & model_types == "TransformerEstimator")
# col_idx_all <-
#   which(!is.na(model_types))
#
# col_idx_all <- best_models$row.id
# results_size <-
#   data.frame(
#     model.type = character(),
#     number.models = integer(),
#     MASE = double(),
#     sMAPE = double()
#   )
# for (num_samples in c(5:length(col_idx_all))) {
#   col_idx <- sample(col_idx_all, size = num_samples)
#   errs <- ensemble_fcasts(col_idx)
#   result <-
#     data.frame(number.models = length(col_idx),
#                sMAPE   = errs[['sMAPE']],
#                MASE    = errs[['MASE']])
#
#   results_size <- rbind(results_size, result)
# }
# write.csv(results_size,
#           file = paste0(data_set, "-", version, "-ensemble_err_vs_size.csv"),
#           row.names = FALSE,
#           quote = FALSE)
# print(results_size)
#
# gg <-
#   ggplot(results_size, aes(x = number.models, y = sMAPE)) +
#   geom_point(size = 0.1) +
#   geom_smooth(size = 0.5, se = FALSE) +
#   labs(title = "Forecast MASE versus number of ensembled models",
#        x = "Number of models",
#        y = "sMAPE")
# ggsave(paste0(data_set, "-", version, "-ensemble_err_vs_size.png"), gg, width = 8, height = 6)

# Timings
tibble(
  model.type = best_models$type,
  book.time = ymd_hms(mydata$book_time[best_models$row.id], tz = "EET"),
  refresh.time = ymd_hms(mydata$refresh_time[best_models$row.id], tz = "EET")
) %>%
  mutate(duration = refresh.time - book.time) %>%
  group_by(model.type) %>%
  summarise(n(), mean(duration), sd(duration)) ->
  timings

write.csv(
  timings,
  file = paste0(data_set, "-", version, "-timings.csv"),
  row.names = FALSE,
  quote = FALSE
)
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
