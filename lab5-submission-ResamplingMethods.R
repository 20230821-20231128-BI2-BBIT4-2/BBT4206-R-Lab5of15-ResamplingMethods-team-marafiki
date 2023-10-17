#installing all needed libraries 

if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("mlbench")) {
  library(mlbench)
} else {
  install.packages("mlbench", dependencies = TRUE, repos = "https://cloud.r-project.org")
  library(mlbench)
}

# Load the Pima Indians Diabetes dataset
data("PimaIndiansDiabetes")

# Split the dataset into a 75% training set and a 25% testing set
splitIndex <- createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.75, list = FALSE)
Pima_train <- PimaIndiansDiabetes[splitIndex, ]
Pima_test <- PimaIndiansDiabetes[-splitIndex, ]

# Check the dimensions of the training and testing sets
cat("Training set dimensions: ", dim(Pima_train), "\n")
cat("Testing set dimensions: ", dim(Pima_test), "\n")

# Train a Naive Bayes classifier using e1071 library
pima_nb_model <- naiveBayes(diabetes ~ ., data = Pima_train)

# Print the model summary
print(pima_nb_model)

# Use the trained Naive Bayes model to make predictions on the testing dataset
predictions <- predict(pima_nb_model, newdata = Pima_test)

# Create a confusion matrix
confusion_matrix <- table(Actual = Pima_test$diabetes, Predicted = predictions)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Calculate and print accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy: ", accuracy, "\n")

# Create a heatmap of the confusion matrix
heatmap(confusion_matrix, 
        col = colorRampPalette(c("Black", "red"))(10), 
        main = "Confusion Matrix Heatmap", 
        xlab = "Predicted", 
        ylab = "Actual")

# Create a table of predictions vs. actual values
prediction_table <- table(Predicted = predictions, Actual = Pima_test$diabetes)

# Plot the table
print("Prediction Table:")
print(prediction_table)

# Ensure the 'diabetes' variable is numeric in the test dataset
Pima_test$diabetes <- as.numeric(Pima_test$diabetes)

lm_model <- lm(diabetes ~ ., data = Pima_train)
# Use the trained linear regression model to make predictions on the testing dataset
lm_predictions <- predict(lm_model, newdata = Pima_test)

# Evaluate the model's performance
actual_values <- Pima_test$diabetes
mse <- mean((lm_predictions - actual_values)^2)
rmse <- sqrt(mse)

# Print the evaluation results
cat("Mean Squared Error (MSE): ", mse, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")

# Create a data frame with new data for prediction
new_data <- data.frame(
  pregnant = 5,
  glucose = 110,
  pressure = 70,
  triceps = 30,
  insulin = 50,
  mass = 56.0,
  pedigree = 0.324,
  age = 35
  # Make sure to add values for all the independent variables in the same order
)

# Use the trained linear regression model to make predictions on the new data
predictions <- predict(lm_model, newdata = new_data)

# Convert numeric predictions to "neg" or "pos"
predictions_binary <- ifelse(predictions > 0.5, "pos", "neg")

# Print the predicted diabetes outcome
cat("Predicted diabetes outcome for the new data:")
print(predictions_binary)

# Define the control parameters for 10-fold cross-validation with accuracy as the metric
ctrl <- trainControl(
  method = "cv",        # Use 10-fold cross-validation
  number = 10,          # Number of folds
  classProbs = TRUE,    # Enable class probabilities for logistic regression
  summaryFunction = twoClassSummary,  # Use the default summary function for classification
  verboseIter = TRUE,   # Display progress during cross-validation
  savePredictions = "final",  # Save final predictions
  selectionFunction = "best",  # Select the best model based on accuracy
  metric = "Accuracy"  # Evaluation metric (accuracy)
)

# Train a logistic regression model using 10-fold cross-validation
logistic_cv_model <- train(diabetes ~ ., data = PimaIndiansDiabetes, method = "glm", family = "binomial", trControl = ctrl) 

# Print the summary of the cross-validated logistic regression model
print(summary(logistic_cv_model))

# Access the accuracy from the logistic regression model
accuracy <- logistic_cv_model$results$Accuracy

# Print the accuracy
cat("Model Accuracy (10-fold cross-validation):", accuracy, "\n")

# Use the trained linear regression model to make predictions on the testing dataset
lm_predictions <- predict(lm_model, newdata = Pima_test)

# Print the first few predicted values for illustration
cat("Predicted values for the testing dataset:\n")
head(lm_predictions)

# Evaluate the model's performance
actual_values <- Pima_test$diabetes  # The actual target values from the testing dataset
mse <- mean((lm_predictions - actual_values)^2)  # Calculate Mean Squared Error
rmse <- sqrt(mse)  # Calculate Root Mean Squared Error

# Print the evaluation results
cat("Mean Squared Error (MSE): ", mse, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")

# Define the control parameters for 5-fold cross-validation with LDA for classification
ctrl <- trainControl(
  method = "cv",                # Use 5-fold cross-validation
  number = 5,                   # Number of folds
  classProbs = TRUE,            # Enable class probabilities
  summaryFunction = twoClassSummary,  # Use summary function for binary classification
  verboseIter = TRUE,           # Display progress during cross-validation
  savePredictions = "final",    # Save final predictions
  selectionFunction = "best",  # Select the best model
  metric = "ROC"               # Evaluation metric (ROC AUC)
)

# Train an LDA classifier using 5-fold cross-validation
lda_cv_model <- train(diabetes ~ ., data = PimaIndiansDiabetes, method = "lda", trControl = ctrl)

# Print the summary of the cross-validated LDA classifier
print(summary(lda_cv_model))


# Use the trained LDA model to make predictions on the testing dataset
lda_predictions <- predict(lda_cv_model, newdata = Pima_test)

# Print the first few predicted values for illustration
cat("Predicted values for the testing dataset:\n")
head(lda_predictions)

# Evaluate the model's performance
actual_values <- Pima_test$churn  # The actual target values from the testing dataset
confusion_matrix <- table(Actual = actual_values, Predicted = lda_predictions)

# Print the confusion matrix
cat("Confusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy and other classification metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the classification metrics
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")

# Classification: Naive Bayes with Repeated k-fold Cross Validation
ctrl <- trainControl(
  method = "repeatedcv",  # Use repeated k-fold cross-validation
  number = 10,            # Number of folds
  repeats = 5,            # Number of repeats
  summaryFunction = twoClassSummary,  # Use summary function for binary classification
  verboseIter = TRUE,     # Display progress during cross-validation
  savePredictions = "final",  # Save final predictions
  classProbs = TRUE,      # Enable class probabilities
  selectionFunction = "best",  # Select the best model based on ROC AUC
  metric = "ROC"          # Evaluation metric (ROC AUC)
)

# Train a Naive Bayes classifier based on the "diabetes" variable
nb_cv_model <- train(diabetes ~ ., data = PimaIndiansDiabetes, method = "naive_bayes", trControl = ctrl)

# Print the summary of the cross-validated Naive Bayes classifier
print(summary(nb_cv_model))

# Use the trained Naive Bayes model to make predictions on the testing dataset
nb_predictions <- predict(nb_cv_model, newdata = Pima_test)

# Print the first few predicted values for illustration
cat("Predicted values for the testing dataset:\n")
head(nb_predictions)

# Evaluate the model's performance
actual_values <- Pima_test$diabetes  # The actual target values from the testing dataset
confusion_matrix <- table(Actual = actual_values, Predicted = nb_predictions)

# Print the confusion matrix
cat("Confusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy and other classification metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the classification metrics
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")


# Define the number of repetitions and folds
repeats <- 3
folds <- 5

# Define the train control settings for repeated k-fold cross-validation
ctrl <- trainControl(
  method = "repeatedcv",   # Repeated k-fold cross-validation
  number = folds,         # Number of folds
  repeats = repeats,      # Number of repetitions
  verboseIter = TRUE
)

# Train an SVM classifier using the diabetes variable
svm_model <- train(diabetes ~ ., data = PimaIndiansDiabetes, method = "svmRadial",
                   trControl = ctrl)

# Print the summary of the SVM model
print(svm_model)

# Access the accuracy or other metrics
svm_accuracy <- svm_model$results$Accuracy
cat("SVM Model Accuracy:", svm_accuracy, "\n")

# Create an empty vector to store predictions
nb_predictions <- numeric(nrow(PimaIndiansDiabetes))

# Perform LOOCV for the Naive Bayes classifier
for (i in 1:nrow(PimaIndiansDiabetes)) {
  # Create a training dataset excluding the current row (i)
  train_data <- PimaIndiansDiabetes[-i, ]
  
  # Train the Naive Bayes classifier
  nb_model <- naiveBayes(diabetes ~ ., data = train_data)
  
  # Make a prediction on the current row
  nb_predictions[i] <- predict(nb_model, newdata = PimaIndiansDiabetes[i, ])
}

# Evaluate the model's performance
actual_values <- PimaIndiansDiabetes$churn
confusion_matrix <- table(Actual = actual_values, Predicted = nb_predictions)

# Print the confusion matrix
cat("Confusion Matrix:\n")
print(confusion_matrix)

# Calculate accuracy and other classification metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the classification metrics
cat("Accuracy (LOOCV): ", accuracy, "\n")
cat("Precision (LOOCV): ", precision, "\n")
cat("Recall (LOOCV): ", recall, "\n")
cat("F1 Score (LOOCV): ", f1_score, "\n")

