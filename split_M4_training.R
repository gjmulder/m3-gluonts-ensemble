# library(zoo)
# library(tsibble)
# library(parallel)
# library(anytime)

library(M4comp2018)
library(lubridate)
library(jsonlite)
library(forecast)
library(tidyverse)

set.seed(42)
options(warn = 0)
options(width = 1024)

###########################################################################
# Config ####

if (interactive()) {
  prop_tt <- NA # 0.1
  # num4_cores <- 4
} else
{
  prop_tt <- NA
  # num4_cores <- 16
}
# use_parallel <- TRUE #is.na(prop_tt)

###########################################################################
# Preprocess M data ####

if (is.na(prop_tt)) {
  m4_data <- M4
} else {
  m4_data <- sample(M4, prop_tt * length(M4))
}

# # Process m4-info start dates
# m4_info_df <- read_csv("M4-info.csv")
#
# # "1750-01-01 00:00:00"
# fix_start <- function(date_str) {
#   # 18th Century
#   if (nchar(date_str) == 19) {
#     return(date_str)
#   }
#
#   day_str <- substr(date_str, 1, 2)
#   month_str <- substr(date_str, 4, 5)
#   year_str <- substr(date_str, 7, 8)
#
#   if (nchar(date_str) == 13) {
#     # Single digit hour
#     hour_str <- paste0("0", substr(date_str, 10, 10))
#     min_str <- substr(date_str, 12, 13)
#     # Double digit hour
#   } else {
#     hour_str <- substr(date_str, 10, 11)
#     min_str <- substr(date_str, 13, 14)
#   }
#   # print(paste(day_str, month_str, year_str, hour_str, min_str))
#
#   if (as.integer(year_str) > 18) {
#     # 19th Century
#     return(
#       paste0(
#         "19",
#         year_str,
#         "-",
#         month_str,
#         "-",
#         day_str,
#         " ",
#         hour_str,
#         ":",
#         min_str,
#         ":00"
#       )
#     )
#   } else {
#     # 20th Century
#     return(
#       paste0(
#         "20",
#         year_str,
#         "-",
#         month_str,
#         "-",
#         day_str,
#         " ",
#         hour_str,
#         ":",
#         min_str,
#         ":00"
#       )
#     )
#   }
# }
#
# get_date <- function(tt, offset) {
#   iso_date <- date_decimal(as.numeric(time(tt)[offset]))
#   # print(substring(iso_date, 1, 10))
#   print(iso_date)
#   return (substring(iso_date, 1, 10))
# }

tt_to_json <- function(idx, tt_list, type_list) {
  print(min(tt_list[[idx]]))
  json <- (paste0(toJSON(
    list(
      start = "1750-01-01 00:00:00",
      target = tt_list[[idx]],
      feat_static_cat = c(idx, as.numeric(type_list[[idx]]))
      # feat_dynamic_real = matrix(rexp(10 * length(tt_list[[idx]])), ncol =
      #                              10)
    ),
    auto_unbox = TRUE
  ), "\n"))
  return(json)
}

# tt_to_csv <- function(idx, tt_list, type_list, horiz_list) {
#   rows <- list()
#   for (x in 1:(length(tt_list[[idx]])-window_size)) {
#     rows[[x]] <- c(get_date(tt_list[[idx]], x),
#                    paste0("IDX", idx),
#                    as.character(type_list[[idx]]),
#                    tt_list[[idx]][x:(x+window_size)])
#   }
#   df <- as.data.frame(t(as.data.frame(rows)))
#   rownames(df) <- c() #rep(idx, nrow(df))
#   colnames(df) <-
#     c("START_DATE", "IDX", "TYPE", paste0("X", window_size:1), "Y")
#
#   # Denote validate and test rows as the last two prediction horizons, repsectively
#   df["DATA_SPLIT"] <- c(rep("TRAIN", nrow(df)-2*horiz_list[[idx]]),
#                   rep("VALIDATE", horiz_list[[idx]]),
#                   rep("TEST", horiz_list[[idx]]))
#   # if (idx > 198)
#   #   browser()
#
#     return(df)
# }

process_period <- function(period, m4_data, final_mode) {
  print(period)
  m4_period_data <- keep(m4_data, function(tt) tt$period == period)

  # len_m4_period <-
  #   unlist(lapply(m4_period_data, function(tt)
  #     return(length(tt$x))))
  # print(ggplot(tibble(tt_length = len_m4_period)) + geom4_histogram(aes(x = tt_length), bins=100) + ggtitle(period) + scale_x_log10())

  # if (use_parallel) {
  #   m4_data_x_deseason <- mclapply(1:length(m4_data_x), function(idx)
  #     return(deseasonalise(m4_data_x[[idx]], m4_horiz[[idx]])), mc.cores = num4_cores)
  # } else {
  #   m4_data_x_deseason <- lapply(1:length(m4_data_x), function(idx)
  #     return(deseasonalise(m4_data_x[[idx]], m4_horiz[[idx]])))
  # }

  m4_st <-
    lapply(m4_period_data, function(tt)
      return(tt$st))

  m4_type_str <-
    lapply(m4_period_data, function(tt)
      return(tt$type))
  m4_type_levels <-
    levels(as.factor(unlist(m4_type_str)))
  m4_type <-
    lapply(m4_type_str, function(type)
      return(factor(type, levels = m4_type_levels)))

  m4_horiz <-
    lapply(m4_period_data, function(tt)
      return(tt$h))

  ###########################################################################
  # Create tt depending on final_mode ####

  if (validation_mode) {
    dirname <-
      paste0("m4_",
             tolower(period),
             '/')
    dir.create(dirname)

    # train - h
    m4_train <-
      lapply(m4_period_data, function(tt)
        return(subset(tt$x, end = (
          length(tt$x) - tt$h
        ))))

    # train + test
    m4_test <-
      lapply(m4_period_data, function(tt)
        return(tt$x))
  } else {
    dirname <-
      paste0("m4_",
             tolower(period),
             '_all/')
    dir.create(dirname)

    # train
    m4_train <-
      lapply(m4_period_data, function(tt)
        return(tt$x))

    # train + test
    m4_test <-
      lapply(m4_period_data, function(tt)
        return(ts(data = c(tt$x, tt$xx), start = start(tt$x), frequency = frequency(tt$x))))
  }

  ###########################################################################
  # Write JSON train and test data ####

  json <-
    lapply(1:length(m4_train),
           tt_to_json,
           m4_train,
           m4_type)
  dir.create(paste0(dirname, "train"))
  sink(paste0(dirname, "train/data.json"))
  lapply(json, cat)
  sink()

  json <-
    lapply(1:length(m4_test),
           tt_to_json,
           m4_test,
           m4_type)
  dir.create(paste0(dirname, "test"))
  sink(paste0(dirname, "test/data.json"))
  lapply(json, cat)
  sink()

  # ###########################################################################
  # # Write csv train and test data ####
  #
  # dfs <-
  #   lapply(1:length(m4_test),
  #          tt_to_csv,
  #          m4_test,
  #          m4_type,
  #          m4_horiz)
  # df <-  do.call(rbind, dfs)
  # write.csv(df, paste0("windowed37_data.csv"), row.names=FALSE)

  return(length(m4_train))
}

# # Size of in-sample window for generating csv data
# window_size <-37

# In validation mode we split the training data into training and validation data sets
validation_mode <- FALSE

# print("!!!! DOES NOT SUPPORT HOURLY AS get_date() RETURNS DATE STRING, NOT 'yyyy-mm-dd HH:MM:SS' !!!!")
# periods <- as.vector(levels(m4_data[[1]]$period))
periods <- c("Daily")
res <- unlist(lapply(periods, process_period, m4_data, final_mode))
names(res) <- periods
print(res)
print(sum(res))
