library(Mcomp)
# library(parallel)
library(lubridate)
# library(anytime)
library(jsonlite)
# library(forecast)
# library(zoo)
library(tidyverse)
# library(tsibble)

set.seed(42)
options(warn = 0)
options(width = 1024)

###########################################################################
# Config ####

if (interactive()) {
  prop_tt <- NA
  # num3_cores <- 4
} else
{
  prop_tt <- NA
  # num3_cores <- 16
}
# use_parallel <- TRUE #is.na(prop_tt)

###########################################################################
# Preprocess M data ####

if (is.na(prop_tt)) {
  m3_data <- M3
} else {
  m3_data <- sample(M3, prop_tt * length(M3))
}

get_date <- function(tt, offset) {
  iso_date <- date_decimal(as.numeric(time(tt)[offset]))
  return (substring(iso_date, 1, 10))
}

tt_to_json <- function(idx, tt_list, type_list) {
  json <- (paste0(toJSON(
    list(
      start = get_date(tt_list[[idx]], 1),
      target = tt_list[[idx]],
      feat_static_cat = c(idx, as.numeric(type_list[[idx]]))
      # feat_dynamic_real = matrix(rexp(10 * length(tt_list[[idx]])), ncol =
      #                              10)
    ),
    auto_unbox = TRUE
  ), "\n"))
  return(json)
}

tt_to_csv <- function(idx, tt_list, type_list, horiz_list) {
  rows <- list()
  for (x in 1:(length(tt_list[[idx]])-window_size)) {
    rows[[x]] <- c(get_date(tt_list[[idx]], x),
                   paste0("IDX", idx),
                   as.character(type_list[[idx]]),
                   tt_list[[idx]][x:(x+window_size)])
  }
  df <- as.data.frame(t(as.data.frame(rows)))
  rownames(df) <- c() #rep(idx, nrow(df))
  colnames(df) <-
    c("START_DATE", "IDX", "TYPE", paste0("X", window_size:1), "Y")

  # Denote validate and test rows as the last two prediction horizons, repsectively
  df["DATA_SPLIT"] <- c(rep("TRAIN", nrow(df)-2*horiz_list[[idx]]),
                  rep("VALIDATE", horiz_list[[idx]]),
                  rep("TEST", horiz_list[[idx]]))
  # if (idx > 198)
  #   browser()

    return(df)
}

process_period <- function(period, m3_data, final_mode) {
  print(period)
  m3_period_data <- keep(subset(m3_data, period), function(tt) length(tt$x) > 80)

  # len_m3_period <-
  #   unlist(lapply(m3_period_data, function(tt)
  #     return(length(tt$x))))
  # print(ggplot(tibble(tt_length = len_m3_period)) + geom3_histogram(aes(x = tt_length), bins=100) + ggtitle(period) + scale_x_log10())

  # if (use_parallel) {
  #   m3_data_x_deseason <- mclapply(1:length(m3_data_x), function(idx)
  #     return(deseasonalise(m3_data_x[[idx]], m3_horiz[[idx]])), mc.cores = num3_cores)
  # } else {
  #   m3_data_x_deseason <- lapply(1:length(m3_data_x), function(idx)
  #     return(deseasonalise(m3_data_x[[idx]], m3_horiz[[idx]])))
  # }

  m3_st <-
    lapply(m3_period_data, function(tt)
      return(tt$st))

  m3_type_str <-
    lapply(m3_period_data, function(tt)
      return(tt$type))
  m3_type_levels <-
    levels(as.factor(unlist(m3_type_str)))
  m3_type <-
    lapply(m3_type_str, function(type)
      return(factor(type, levels = m3_type_levels)))

  m3_horiz <-
    lapply(m3_period_data, function(tt)
      return(tt$h))

  ###########################################################################
  # Create tt depending on final_mode ####

  if (validation_mode) {
    dirname <-
      paste0("m3_",
             tolower(period),
             '/')
    # train - h
    m3_train <-
      lapply(m3_period_data, function(tt)
        return(subset(tt$x, end = (
          length(tt$x) - tt$h
        ))))

    # train + test
    m3_test <-
      lapply(m3_period_data, function(tt)
        return(tt$x))
  } else {
    dirname <-
      paste0("m3_",
             tolower(period),
             '_all/')
    # train
    m3_train <-
      lapply(m3_period_data, function(tt)
        return(tt$x))

    # train + test
    m3_test <-
      lapply(m3_period_data, function(tt)
        return(ts(data = c(tt$x, tt$xx), start = start(tt$x), frequency = frequency(tt$x))))
  }

  # ###########################################################################
  # # Write JSON train and test data ####
  #
  # json <-
  #   lapply(1:length(m3_train),
  #          tt_to_json,
  #          m3_train,
  #          m3_type)
  # sink(paste0(dirname, "train/data.json"))
  # lapply(json, cat)
  # sink()
  #
  # json <-
  #   lapply(1:length(m3_test),
  #          tt_to_json,
  #          m3_test,
  #          m3_type)
  # sink(paste0(dirname, "test/data.json"))
  # lapply(json, cat)
  # sink()

  ###########################################################################
  # Write csv train and test data ####

  dfs <-
    lapply(1:length(m3_test),
           tt_to_csv,
           m3_test,
           m3_type,
           m3_horiz)
  df <-  do.call(rbind, dfs)
  write.csv(df, paste0("windowed37_data.csv"), row.names=FALSE)

  return(length(m3_train))
}

# Size of in-sample window for generating csv data
window_size <-37

# In validation mode we split the training data into training and validation data sets
validation_mode <- FALSE

print("!!!! DOES NOT SUPPORT HOURLY AS get_date() RETURNS DATE STRING, NOT 'yyyy-mm-dd HH:MM:SS' !!!!")
# periods <- as.vector(levels(m3_data[[1]]$period))
periods <- c("monthly")
res <- unlist(lapply(periods, process_period, m3_data, final_mode))
names(res) <- periods
print(res)
print(sum(res))
