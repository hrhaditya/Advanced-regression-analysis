# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)

# Read in the data
data_url <- "https://raw.githubusercontent.com/hrhaditya/Advanced-regression-analysis/main/TechSales_Reps_Data(1).csv"
data <- read_csv(data_url)

# Filter data to include only software product group employees with a college degree
filtered_data <- data %>%
  filter(Business == "Software" & College == "Yes")

# Create the dummy target variable representing NPS ≥ 9
filtered_data$NPS_high <- as.integer(filtered_data$NPS >= 9)

# EDA: Visualizing the distribution of NPS_high in the filtered data
nps_high_plot <- ggplot(filtered_data, aes(x = NPS_high)) + 
  geom_bar() +
  labs(title = "Distribution of NPS_high", x = "NPS_high", y = "Count")

print(nps_high_plot)

# EDA: Visualizing relationships between predictor variables and NPS_high
age_nps_high_plot <- ggplot(filtered_data, aes(x = Age, y = NPS_high)) + 
  geom_jitter(alpha = 0.5) +
  labs(title = "Age vs NPS_high", x = "Age", y = "NPS_high")

print(age_nps_high_plot)

# Split the data into training and test sets
set.seed(123)
training_indices <- createDataPartition(filtered_data$NPS_high, p = 0.8, list = FALSE)
training_data <- filtered_data[training_indices,]
test_data <- filtered_data[-training_indices,]

# Perform k-fold cross-validation
set.seed(123)
k <- 10
folds <- createFolds(filtered_data$NPS_high, k = k)
accuracy_list <- vector("numeric", length = k)

for (i in 1:k) {
  training_data <- filtered_data[-folds[[i]],]
  test_data <- filtered_data[folds[[i]],]
  
  
  # Build the Random Forest model
  rf_model <- randomForest(as.factor(NPS_high) ~ Age + Female + Years + Personality + Certficates + Feedback + Salary, 
                           data = training_data, ntree = 500)
  
  # Predict the test data and calculate the model's accuracy
  test_data$predicted_label <- predict(rf_model, newdata = test_data, type = "response")
  accuracy_list[i] <- mean(test_data$predicted_label == as.factor(test_data$NPS_high))
  
}

# Calculate the average accuracy from k-fold cross-validation
avg_accuracy <- mean(accuracy_list)
print(avg_accuracy)

# Build the logistic regression model
model <- glm(NPS_high ~ Age + Female + Years + Personality + Certficates + Feedback + Salary, 
             data = training_data, family = binomial(link = "logit"))

# Model summary
summary(model)

# Predict the test data and calculate the model's accuracy
test_data$predicted <- predict(model, newdata = test_data, type = "response")
test_data$predicted_label <- as.integer(test_data$predicted >= 0.5)
accuracy <- mean(test_data$predicted_label == test_data$NPS_high)
accuracy

# Define a tuning grid for the random forest model
tuning_grid <- expand.grid(.mtry = seq(from = 2, to = ncol(training_data) - 1, by = 1))

# Set up the control for hyperparameter tuning
control <- trainControl(method = "cv", number = 5, search = "grid")

# Perform hyperparameter tuning
set.seed(123)
tuned_rf_model <- train(as.factor(NPS_high) ~ Age + Female + Years + Personality + Certficates + Feedback + Salary,
                        data = training_data,
                        method = "rf",
                        metric = "Accuracy",
                        trControl = control,
                        tuneGrid = tuning_grid)

# Print the best hyperparameters
print(tuned_rf_model$bestTune)

# Train the final model using the optimal mtry value
optimal_mtry <- tuned_rf_model$bestTune$mtry
final_rf_model <- randomForest(as.factor(NPS_high) ~ Age + Female + Years + Personality + Certficates + Feedback + Salary,
                               data = training_data,
                               ntree = 500,
                               mtry = optimal_mtry)

# Evaluate the model on the test set
test_data$predicted_label <- predict(final_rf_model, newdata = test_data)
accuracy <- mean(test_data$predicted_label == as.factor(test_data$NPS_high))
print(accuracy)

# Print the best hyperparameters
cat("Best mtry value:", tuned_rf_model$bestTune$mtry, "\n")

# Train the final model using the optimal mtry value
optimal_mtry <- tuned_rf_model$bestTune$mtry
final_rf_model <- randomForest(as.factor(NPS_high) ~ Age + Female + Years + Personality + Certficates + Feedback + Salary,
                               data = training_data,
                               ntree = 500,
                               mtry = optimal_mtry)

# Evaluate the model on the test set
test_data$predicted_label <- predict(final_rf_model, newdata = test_data)
accuracy <- mean(test_data$predicted_label == as.factor(test_data$NPS_high))
cat("Test set accuracy:", accuracy, "\n")

# Data Visualization
# Load libraries
library(ggplot2)
library(gridExtra)

# Plot 1: Age vs NPS_high
p1 <- ggplot(training_data, aes(x = Age, y = NPS_high)) +
  geom_jitter(alpha = 0.5) +
  geom_smooth(method = "loess") +
  ggtitle("Age vs NPS_high") +
  xlab("Age") +
  ylab("NPS_high")

# Plot 2: Years vs NPS_high
p2 <- ggplot(training_data, aes(x = Years, y = NPS_high)) +
  geom_jitter(alpha = 0.5) +
  geom_smooth(method = "loess") +
  ggtitle("Years vs NPS_high") +
  xlab("Years") +
  ylab("NPS_high")

# Plot 3: Certificates vs NPS_high
p3 <- ggplot(training_data, aes(x = Certficates, y = NPS_high)) +
  geom_jitter(alpha = 0.5) +
  geom_smooth(method = "loess") +
  ggtitle("Certificates vs NPS_high") +
  xlab("Certificates") +
  ylab("NPS_high")

# Plot 4: Feedback vs NPS_high
p4 <- ggplot(training_data, aes(x = Feedback, y = NPS_high)) +
  geom_jitter(alpha = 0.5) +
  geom_smooth(method = "loess") +
  ggtitle("Feedback vs NPS_high") +
  xlab("Feedback") +
  ylab("NPS_high")

# Arrange the plots in a grid
grid.arrange(p1, p2, p3, p4, ncol = 2)

# Load ggplot2 library
library(ggplot2)

# Generate random data
data <- data.frame(values = rnorm(1000))

# Create a histogram
ggplot(data, aes(x = values)) + geom_histogram(fill = "lightblue", color = "black") + labs(title = "Histogram", x = "Values", y = "Frequency")

# Load ggplot2 library
library(ggplot2)

# Create sample data
data <- data.frame(categories = c("A", "B", "C", "D"), values = c(10, 30, 50, 20))

# Create a bar graph
ggplot(data, aes(x = categories, y = values)) + geom_bar(stat = "identity", fill = "lightblue", color = "black") + labs(title = "Bar Graph", x = "Categories", y = "Values")
