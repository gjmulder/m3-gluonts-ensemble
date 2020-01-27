library(mongolite)
library(jsonlite)
library(matrixStats)
library(lubridate)
library(gtools)
library(dplyr)
# library(ggpubr)
library(ggplot2)

set.seed(42)

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
  stopifnot(!is.na(insample),
            !is.na(outsample),
            !is.na(forecasts),
            !is.na(frq))

  # Used to estimate MASE
  forecastsNaiveSD <- rep(NA, frq)
  for (j in (frq + 1):length(insample)) {
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j - frq])
  }
  masep <- mean(abs(insample - forecastsNaiveSD), na.rm = TRUE)

  outsample <-
    as.numeric(outsample)
  forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample - forecasts)) / masep

  stopifnot(!is.na(mase),!is.nan(mase))
  return(mase)
}

ensemble_fcasts <- function(col_idx) {
  # print(col_idx)
  fcasts_df <- mongo_data$result$err_metrics$y_hats[col_idx, ]
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
        model_col_idx <-
          model_set %>%
          filter(type == model_type) %>%
          pull(row.id)
        model_col_idx <-
          sample(model_col_idx, size = 88, replace = FALSE)
        col_idx <- c(col_idx, model_col_idx)
      }
      errs <- ensemble_fcasts(col_idx)
      result <-
        data.frame(
          model.types = paste0(comb[idx2, ], collapse = "+"),
          model.type.count = length(comb[idx2, ]),
          total.num.models = length(col_idx),
          MASE    = errs[['MASE']],
          sMAPE   = errs[['sMAPE']]
        )
      results_comb <- rbind(results_comb, result)
    }
  }
  return(results_comb)
}

####################################################################################
# Get forecasts from MongoDB and test data from .json

dataset_version <- Sys.getenv(c("DATASET", "VERSION"))
data_set <- "m3_yearly" #as.character(dataset_version[1])
version <- "ensemble-v01" #as.character(dataset_version[2])

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
mongo_data <- con$find()
rm(con)
gc()
model_type_loss <-
  tibble(
    row.id = c(1:length(mongo_data$result$cfg$model$type)),
    type = mongo_data$result$cfg$model$type,
    loss = mongo_data$result$loss
  ) %>%
  na.omit %>%
  filter(type != "DeepFactorEstimator" &
           type != "GaussianProcessEstimator")

res <-
  # readLines(paste0("/var/tmp/", data_set, "_all/test/data.json"))
  readLines("/home/mulderg/Work/plos1-m3/m3_yearly_all/test/data.json")
# readLines("/home/mulderg/Work/plos1-m3/m3_monthly_all/test/data.json")

####################################################################################
# Timings

tibble(
  model.type = model_type_loss$type,
  book.time = ymd_hms(mongo_data$book_time[model_type_loss$row.id], tz = "EET"),
  refresh.time = ymd_hms(mongo_data$refresh_time[model_type_loss$row.id], tz = "EET")
) %>%
  mutate(duration = refresh.time - book.time) %>%
  na.omit %>%
  group_by(model.type) %>%
  summarise(n(), mean(duration), sd(duration)) ->
  timings
print(timings)

####################################################################################
# Model combinations

# for (n in c(70, 80, 90, 100)) {
# model_type_loss %>%
# filter(type == "DeepAREstimator") ->
# na.omit %>%
# group_by(type) %>%
# sample_n(n, replace = FALSE) ->
# top_n(-n, loss) ->
# top_frac(-n/100, loss) ->
# best_models
# print(best_models)

build_ensemble(model_type_loss) %>%
  na.omit %>%
  mutate(model.types = gsub("Estimator", "", model.types)) %>%
  mutate(model.count.fact = factor(paste0(
    total.num.models, " [", model.type.count, "]"
  ))) %>%
  arrange(desc(sMAPE)) ->
  results_comb
# results_comb$n.star.models <- rep(n, nrow(results_comb))
print(tail(results_comb[, c("model.types", "total.num.models", "MASE", "sMAPE")], 100))
#}

gg_comb <-
  results_comb %>%
  mutate(model.count.fact = reorder(model.count.fact, total.num.models)) %>%
  ggplot(aes(x = model.count.fact, y = sMAPE)) +
  geom_violin() +
  stat_summary(
    fun.y = median,
    geom = "point",
    size = 1,
    color = "red"
  ) +
  labs(title = "Forecast error versus number of ensembled combinations of models",
       x = "Ensemble Size [Number of model types per ensemble]")

if (interactive()) {
  print(gg_comb)
}

####################################################################################
# Sampling

model_type_loss %>%
  pull(row.id) ->
  col_idx_all

results_size <-
  data.frame(
    model.type = character(),
    number.models = integer(),
    MASE = double(),
    sMAPE = double()
  )
for (num_samples in c(25:length(col_idx_all))) {
  col_idx <- sample(col_idx_all, size = num_samples, replace = FALSE)
  errs <- ensemble_fcasts(col_idx)
  result <-
    data.frame(
      number.models = length(col_idx),
      sMAPE   = errs[['sMAPE']],
      MASE    = errs[['MASE']]
    )
  results_size <- rbind(results_size, result)
}

gg_size <-
  ggplot(results_size, aes(x = number.models, y = sMAPE)) +
  geom_point(size = 0.1) +
  geom_smooth(size = 0.5, se = FALSE) +
  # stat_regline_equation(
  #   aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~~")),
  #   label.x = 150,
  #   label.y = 15.0,
  #   colour = "red",
  #   na.rm = TRUE
  # )
  labs(title = "Forecast error versus number of ensembled models",
       x = "Number of models")

if (!interactive()) {
  options(width = 1000)

  write.csv(
    timings,
    file = paste0(data_set, "-", version, "-timings.csv"),
    row.names = FALSE,
    quote = FALSE
  )

  write.csv(
    results_comb,
    file = paste0(data_set, "-", version, "-ensemble_combination_results.csv"),
    row.names = FALSE,
    quote = FALSE
  )

  ggsave(
    paste0(data_set, "-", version, "-ensemble_combination_results.png"),
    gg_comb,
    width = 8,
    height = 6
  )

  write.csv(
    results_size,
    file = paste0(data_set, "-", version, "-ensemble_err_vs_size.csv"),
    row.names = FALSE,
    quote = FALSE
  )

  ggsave(
    paste0(data_set, "-", version, "-ensemble_err_vs_size.png"),
    gg_size,
    width = 8,
    height = 6
  )

} else {
  print(gg_size)
}

# for (model_type in uniq_model_types) {
#   col_idx <- which(model_types == model_type)
#   MASEs <- tibble(MASE = mongo_data$result$err_metrics$validate$MASE[col_idx])
#   gg <-
#     ggplot(MASEs, aes(x = MASE)) +
#     geom_histogram(bins = 20) +
#     labs(title = paste0("Distribution of ", model_type, " MASEs"),
#          x = "MASE",
#          y = "Count")
#   print(gg)
# }
