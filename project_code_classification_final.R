# ================================
# Load Required Packages
# ================================

# Use the 'pacman' package for streamlined package management
if (!require("pacman"))
  install.packages("pacman")

pacman::p_load(
  tidyverse, gridExtra, cluster, factoextra, tidyr, Rtsne, dbscan, readr,
  umap, mclust, randomForest, caret, plotly, zoo, lubridate, data.table,
  naivebayes, pROC
)

# ================================
# Define Utility Functions
# ================================

# Normalize Function: Scales data to a range of [0, 1]
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# ================================
# Define Feature List for Clustering
# ================================

numeric_columns <- c(
  'SPX', 'GLD', 'USO', 'SLV', 'EUR.USD',

  # Lag features
  'lag_1_SPX', 'lag_1_GLD', 'lag_1_USO', 'lag_1_SLV', 'lag_1_EUR.USD',
  'lag_2_SPX', 'lag_2_GLD', 'lag_2_USO', 'lag_2_SLV', 'lag_2_EUR.USD',
  'lag_3_SPX', 'lag_3_GLD', 'lag_3_USO', 'lag_3_SLV', 'lag_3_EUR.USD',

  # Rolling averages
  'roll_avg_3_SPX', 'roll_avg_5_SPX', 'roll_avg_10_SPX',
  'roll_avg_3_SLV', 'roll_avg_5_SLV', 'roll_avg_10_SLV',
  'roll_avg_3_USO', 'roll_avg_5_USO', 'roll_avg_10_USO',
  'roll_avg_3_GLD', 'roll_avg_5_GLD', 'roll_avg_10_GLD',
  'roll_avg_3_EUR.USD', 'roll_avg_5_EUR.USD', 'roll_avg_10_EUR.USD',

  # Rolling standard deviations
  'roll_sd_3_SPX', 'roll_sd_5_SPX', 'roll_sd_10_SPX',
  'roll_sd_3_SLV', 'roll_sd_5_SLV', 'roll_sd_10_SLV',
  'roll_sd_3_USO', 'roll_sd_5_USO', 'roll_sd_10_USO',
  'roll_sd_3_GLD', 'roll_sd_5_GLD', 'roll_sd_10_GLD',
  'roll_sd_3_EUR.USD', 'roll_sd_5_EUR.USD', 'roll_sd_10_EUR.USD',

  # Momentum (rate of change) features
  'roc_SPX', 'roc_SLV', 'roc_USO', 'roc_GLD', 'roc_EUR.USD',

  # Normalized features
  'norm_SPX', 'norm_USO', 'norm_SLV', 'norm_EUR.USD'
)

# ================================
# Load Dataset
# ================================

# Read and preprocess the dataset
df <- read.csv("D:\\Lecture Notes\\Fall 2024\\Data Mining\\Project\\gld_price_data.csv")

# ================================
# Preprocess Data
# ================================

# Convert Date column to Date format and filter out missing dates
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df <- filter(df, !is.na(df$Date))

# Add temporal features: day, month, and year for visualizations
df <- df |>
  mutate(
    day = day(Date),
    month = month(Date),
    year = year(Date)
  )

# ================================
# Feature Engineering
# ================================

# Create lag features for predictions
for (i in 1:3) {
  df <- df |>
    mutate(across(
      c(SPX, GLD, USO, SLV, EUR.USD),
      ~lag(., n = i),
      .names = "lag_{i}_{col}"
    ))
}

# Add rolling averages
rolling_windows <- c(3, 5, 10)
for (k in rolling_windows) {
  df <- df |>
    mutate(across(
      c(SPX, GLD, USO, SLV, EUR.USD),
      ~rollmean(., k = k, fill = NA, align = "right"),
      .names = "roll_avg_{k}_{col}"
    ))
}

# Add rolling standard deviations
for (k in rolling_windows) {
  df <- df |>
    mutate(across(
      c(SPX, GLD, USO, SLV, EUR.USD),
      ~rollapply(., width = k, FUN = sd, fill = NA, align = "right"),
      .names = "roll_sd_{k}_{col}"
    ))
}

# Add momentum (rate of change) features
df <- df |>
  mutate(across(
    c(SPX, GLD, USO, SLV, EUR.USD),
    ~(. - lag(.)) / lag(.),
    .names = "roc_{col}"
  ))

# Normalize the dependent features for SVM
df <- df |>
  mutate(across(
    c(SPX, USO, SLV, EUR.USD),
    ~normalize(.),
    .names = "norm_{col}"
  ))

# Add gold label for classification: "Up" if price increases, "Down" otherwise
df <- df |>
  mutate(
    gold_label = case_when(
      lead(GLD) - GLD >= 0 ~ "Up",
      lead(GLD) - GLD < 0 ~ "Down"
    ),
    gold_label = lag(gold_label),
    gold_label = ifelse(is.na(gold_label), "Up", gold_label)
  )

# Write the preprocessed dataset to a CSV file
# write.csv(df, "D:\\Lecture Notes\\Fall 2024\\Data Mining\\Project\\gold_price_dataset_preprocessed.csv")

# ================================
# Split Data into Training and Testing Sets
# ================================

# Ensure target variable is a factor
df <- df |>
  mutate(gold_label = as.factor(gold_label)) |>
  select(-Date, -day, -month, -year, -GLD)

# Split data: 80% training, 20% testing
set.seed(6366)
train_index <- createDataPartition(df$gold_label, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Remove any rows with missing values
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# ================================
# Random Forest Model
# ================================

setDT(train_data)
setDT(test_data)

# Define variables to use in Random Forest
vars_for_RF <- c(
  "lag_1_USO", "lag_1_SLV", "lag_1_EUR.USD", "lag_1_SPX", "gold_label",
  "roc_SLV", "roc_USO", "roc_EUR.USD", "roc_SPX",
  "roll_avg_5_SPX", "roll_avg_5_SLV", "roll_avg_5_EUR.USD", "roll_avg_5_USO",
  "roll_sd_5_SLV", "roll_sd_5_USO", "roll_sd_5_SPX", "roll_sd_5_EUR.USD"
)

vars_for_RF <- intersect(vars_for_RF, colnames(train_data))

# Hyperparameter tuning grid
tune_grid <- expand.grid(
  .mtry = c(4, 6, 8),
  .ntree = c(100, 300, 500, 700, 900)
)

# Initialize empty dataframe for results
tuned_results <- data.frame()

# Train models with all combinations of mtry and ntree
for (i in 1:nrow(tune_grid)) {
  # Get the current values of mtry and ntree
  current_mtry <- tune_grid[i, ".mtry"]
  current_ntree <- tune_grid[i, ".ntree"]

  # Train the Random Forest model
  rf_model <- randomForest(
    gold_label ~ .,
    data = train_data[, ..vars_for_RF],
    mtry = current_mtry,
    ntree = current_ntree,
    importance = TRUE
  )

  # Calculate OOB error and store results
  tuned_results <- rbind(
    tuned_results,
    data.frame(
      mtry = current_mtry,
      ntree = current_ntree,
      OOBError = rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]
    )
  )
}
# Print tuning results
print(tuned_results)

# Select the best hyperparameters
best_combination <- tuned_results[which.min(tuned_results$OOBError), ]
print(paste("Best mtry:", best_combination$mtry,
            "Best ntree:", best_combination$ntree))

# Train final model with best hyperparameters
rf_model_best <- randomForest(
  gold_label ~ .,
  data = train_data[, ..vars_for_RF],
  mtry = best_combination$mtry,
  ntree = best_combination$ntree,
  importance = TRUE
)

# Print final model summary and variable importance plot
print(rf_model_best)
varImpPlot(rf_model_best)

# Make predictions and evaluate performance
predictions <- predict(rf_model_best, test_data)
confusion_matrix <- confusionMatrix(predictions, test_data$gold_label)
print(confusion_matrix)

rf_roc <- roc(test_data$gold_label, as.numeric(predictions == "Up"))
plot(rf_roc, col = "black", main = "RF ROC Curve")

# ================================
# Support Vector Machine (SVM)
# ================================

# Define variables for SVM model
vars_for_SVM <- c(
  "lag_1_USO", "lag_1_SLV", "lag_1_EUR.USD", "lag_1_SPX", "gold_label",
  "roc_SLV", "roc_USO", "roc_EUR.USD", "roc_SPX",
  "roll_avg_5_SPX", "roll_avg_5_SLV", "roll_avg_5_EUR.USD", "roll_avg_5_USO",
  "roll_sd_5_SLV", "roll_sd_5_USO", "roll_sd_5_SPX", "roll_sd_5_EUR.USD",
  "norm_SPX", "norm_USO", "norm_SLV", "norm_EUR.USD"
)

# Define grid for hyperparameter tuning
svm_grid <- expand.grid(
  C = c(0.1, 1, 10, 100),  # Regularization parameter
  sigma = c(0.01, 0.1, 1)  # RBF kernel width
)

# Train SVM model with grid search
svm_model <- train(
  gold_label ~ .,
  data = train_data[, ..vars_for_SVM],
  method = "svmRadial",
  tuneGrid = svm_grid,
  trControl = trainControl(method = "cv", number = 5)  # 5-fold cross-validation
)

# Print the best SVM model and parameters
print(svm_model)

# Make predictions with the best SVM model
svm_predictions <- predict(svm_model, test_data)

# Evaluate the SVM model performance
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$gold_label)

# Print confusion matrix and performance metrics
print(svm_conf_matrix)

# Plot ROC curve for SVM
svm_roc <- roc(test_data$gold_label, as.numeric(svm_predictions == "Up"))
plot(svm_roc, col = "blue", main = "SVM ROC Curve")

# ================================
# Naive Bayes Model
# ================================

# Define variables for Naive Bayes model
vars_for_NB <- c(
  "lag_1_SPX", "lag_1_GLD", "lag_1_USO", "lag_1_SLV", "lag_1_EUR.USD",
  "roll_avg_5_SPX", "roll_avg_5_GLD", "roll_avg_5_SLV",
  "roc_SPX", "roc_GLD", "roc_USO",
  "roll_sd_5_SPX", "roll_sd_5_GLD",
  "gold_label"
)

# Define grid for hyperparameter tuning
nb_grid <- expand.grid(
  laplace = c(0, 1),          # Laplace smoothing
  usekernel = c(TRUE, FALSE), # Use kernel density estimation
  adjust = c(1, 1.5)          # Bandwidth adjustment for kernel
)

# Train Naive Bayes model with grid search
nb_model <- train(
  gold_label ~ .,
  data = train_data[, ..vars_for_NB],
  method = "naive_bayes",
  tuneGrid = nb_grid,
  trControl = trainControl(method = "cv", number = 5)  # 5-fold cross-validation
)

# Print the best Naive Bayes model and parameters
print(nb_model)

# Make predictions with the best Naive Bayes model
nb_predictions <- predict(nb_model, test_data)

# Evaluate the Naive Bayes model performance
nb_conf_matrix <- confusionMatrix(nb_predictions, test_data$gold_label)

# Print confusion matrix and performance metrics
print(nb_conf_matrix)

# Plot ROC curve for Naive Bayes
nb_roc <- roc(test_data$gold_label, as.numeric(nb_predictions == "Up"))
plot(nb_roc, col = "red", main = "Naive Bayes ROC Curve")

# ================================
# Visualize Confusion Matrices
# ================================

# Function to create a confusion matrix heatmap
plot_confusion_matrix <- function(conf_matrix, title) {
  data <- as.data.frame(as.table(conf_matrix$table))
  colnames(data) <- c("Actual", "Predicted", "Freq")

  ggplot(data, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_gradient(low = "red", high = "purple") +
    labs(title = title, fill = "Frequency") +
    theme_minimal()
}

# Random Forest confusion matrix
rf_matrix <- confusionMatrix(predictions, test_data$gold_label)
rf_plot <- plot_confusion_matrix(rf_matrix, "Random Forest Confusion Matrix")

# SVM confusion matrix
svm_plot <- plot_confusion_matrix(svm_conf_matrix, "SVM Confusion Matrix")

# Naive Bayes confusion matrix
nb_plot <- plot_confusion_matrix(nb_conf_matrix, "Naive Bayes Confusion Matrix")

# Print confusion matrix heatmaps
print(rf_plot)
print(svm_plot)
print(nb_plot)

