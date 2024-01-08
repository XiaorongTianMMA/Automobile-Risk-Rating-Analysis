# Xiaorong Tian Final Project
auto = read.csv('C:/Users/95675/OneDrive/桌面/R/Final Project/Dataset 5 — Automobile data_Processed.csv')
attach(auto)

auto$make=as.factor(auto$make)
auto$fuel_type=as.factor(auto$fuel_type)
auto$aspiration=as.factor(auto$aspiration)
auto$num_of_doors=as.factor(auto$num_of_doors)
auto$body_style=as.factor(auto$body_style)
auto$drive_wheels=as.factor(auto$drive_wheels)
auto$engine_type=as.factor(auto$engine_type)
auto$fuel_system=as.factor(auto$fuel_system)
auto$num_of_cylinders=as.factor(auto$num_of_cylinders)
# Part 1: Exploratory Data Description
# 1.1. Key indicators distribution
# Price
library(ggplot2)
ggplot(auto, aes(x = price)) + 
  geom_histogram(binwidth = 500, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Car Prices", 
       x = "Price", 
       y = "Count")

# Horsepower
# Load ggplot2 package
library(ggplot2)

# Create a histogram of the horsepower variable
ggplot(auto, aes(x = horsepower)) + 
  geom_histogram(binwidth = 10, fill = "blue", color = "black") + 
  theme_minimal() +
  labs(title = "Distribution of Horsepower", 
       x = "Horsepower", 
       y = "Count")

# 1.2. Correlation between predictors
library(dplyr)
library(ggplot2)
library(reshape2)

selected_auto <- auto %>%
  select(normalized_losses, wheel_base, length, width, height, curb_weight,
         engine_size, bore, stroke, horsepower,
         compression_ratio, peak_rpm, city_mpg, highway_mpg, price, 
         Power_to_Weight_Ratio, Engine_Efficiency, Total_Displacement, 
         Length_to_Width_Ratio, Length_to_Height_Ratio, Width_to_Height_Ratio)

# Calculate the correlation matrix
cor_matrix <- cor(selected_auto, use = "complete.obs")  # using 'complete.obs' to handle missing values

# Melt the correlation matrix for ggplot
melted_cor_matrix <- melt(cor_matrix)

# Plotting the matrix graph
ggplot(data = melted_cor_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) +
  coord_fixed()
print(cor_matrix)



# Part 1: Logistic Regression
library(dplyr)
library(caret)
library(dplyr)

# Convert categorical variables to numeric using one-hot encoding
auto_categorical = model.matrix(~ make + fuel_type + aspiration + num_of_doors + body_style + drive_wheels + engine_type + fuel_system + num_of_cylinders - 1, data = auto)
auto_numeric = auto[, sapply(auto, is.numeric)]
# Standardize the data
auto_combined = cbind(auto_numeric, auto_categorical)

# Standardize the combined data
auto_scaled = scale(auto_combined)
# Perform PCA
pca_result = prcomp(auto_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA
summary(pca_result)
single_level_factors = sapply(auto, function(x) is.factor(x) && length(levels(x)) < 2)

# Print out the names of these variables
cat("Single-level factors:", names(single_level_factors[single_level_factors]), "\n")

# Scree plot to help decide the number of components to retain
scree_plot = data.frame(Comp = 1:length(pca_result$sdev), Var = pca_result$sdev^2)
ggplot(scree_plot, aes(x = Comp, y = Var)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Scree Plot", x = "Principal Component", y = "Variance Explained")

# Scree plot to help decide the number of components to retain
scree_plot = data.frame(Comp = 1:length(pca_result$sdev), Var = pca_result$sdev^2)
ggplot(scree_plot, aes(x = Comp, y = Var)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Scree Plot", x = "Principal Component", y = "Variance Explained")

# Load the ggfortify package
library(ggfortify)
autoplot(pca_result, data = auto, loadings = TRUE, loadings.label = TRUE, loadings.label.size = 3)

colors = ifelse(auto$symboling == 3, "orange", ifelse(auto$symboling == -1, "blue", "transparent"))

# Create PCA plot with points colored based on symboling values
p = autoplot(pca_result, data = auto, loadings = TRUE,
              col = colors, loadings.label = TRUE) +
  scale_colour_manual(values=c("orange", "blue", "transparent")) +
  theme(legend.position="top") # To hide legend if not needed

# Print the plot
print(p)
# Extract the first three principal components
pca_scores = pca_result$x[, 1:3]

# Create a new data frame with the PCA scores
auto_pca = data.frame(auto, PCA1 = pca_scores[,1], PCA2 = pca_scores[,2], PCA3 = pca_scores[,3])

# Create the new binary variable for symboling
auto_pca$risk_category = ifelse(auto_pca$symboling >= 1 & auto_pca$symboling <= 3, "low_risk", "high_risk")

# Convert the new variable to a binary numeric format
auto_pca$risk_category_numeric = ifelse(auto_pca$risk_category == "low_risk", 1, 0)

# Splitting the dataset into training and testing sets
set.seed(123)  # For reproducibility
indexes = createDataPartition(auto_pca$risk_category_numeric, p = 0.8, list = FALSE)
train_data = auto_pca[indexes, ]
test_data = auto_pca[-indexes, ]

# Building the logistic regression model using the first three principal components
logit_model = glm(risk_category_numeric ~ PCA1 + PCA2 + PCA3, data = train_data, family = binomial())

# Summary of the model
model_summary = summary(logit_model)

# Print the summary to the console
print(model_summary)

# Perform ANOVA to test the significance of the predictors
anova_logit_model = anova(logit_model, test="Chisq")

# Print the ANOVA results to the console
print(anova_logit_model)

# Predicting on test data
predictions = predict(logit_model, newdata = test_data, type = "response")
predicted_class = ifelse(predictions > 0.5, 1, 0)

# Load the caret package for confusion Matrix
library(caret)

# Evaluating model performance
confusionMatrix(factor(predicted_class), factor(test_data$risk_category_numeric))


# Part 2: Random Forest with initial data and processed data with new columns
auto$symboling=as.factor(auto$symboling)
library(randomForest)
myforest2=randomForest(symboling~normalized_losses+make+fuel_type+aspiration+num_of_doors+body_style+drive_wheels+wheel_base+length+width+height+curb_weight+engine_type+num_of_cylinders+engine_size+fuel_system+bore+stroke+compression_ratio+horsepower+peak_rpm+city_mpg+highway_mpg+price+Power_to_Weight_Ratio+Engine_Efficiency+Total_Displacement+Length_to_Width_Ratio+Length_to_Height_Ratio+Width_to_Height_Ratio, ntree=2000, data=auto, importance=TRUE,  na.action = na.omit)
myforest2
importance(myforest2)


