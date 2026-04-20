library(readxl)
library(writexl)
library(tidyverse)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(corrplot)
library(ggcorrplot)

set.seed(123)   # For reproducibility)

# ---------------------------------------------------------
# 1. Load the Data

# In this section, we load the original automobile dataset from Excel and briefly inspect its structure. The dataset contains information such as price, age, mileage, fuel type, color, engine size, and multiple binary equipment features.
# ---------------------------------------------------------

# 1.1 Read the Excel file (original case data)
data <- read_excel("C:/Users/divya/OneDrive/Projects/Capstone/W28593-XLS-ENG.xlsx")

# 1.2 Quick structure check
glimpse(data)
summary(data)

# ---------------------------------------------------------
# 2. Basic Cleaning & Renaming

# We first make column names syntactically safe, check for missing values.
# ---------------------------------------------------------

# 2.1 Make safe column names (no spaces, no special characters)
df <- data
names(df) <- make.names(names(df))

# Check names once
names(df)

# 2.2 Check missing values
colSums(is.na(df))

# Remove rows with any missing values / NA
# (none in this dataset, but kept for robustness)
df <- na.omit(df)

# ---------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)

# We examine correlations among numerical variables to understand how price relates to age, mileage, horsepower, engine size, and weight.

# The scatter plot below shows how price varies with horsepower, with a fitted linear trend.
# ---------------------------------------------------------

# 3.1 Summary statistics
summary(df)

# 3.2 Correlation plot for numeric variables
numeric_vars <- df %>% select_if(is.numeric)
corr_matrix <- cor(numeric_vars)

ggcorrplot(
  corr_matrix,
  type      = "lower",
  lab       = FALSE,   # set TRUE if you want numeric values
  title = "Correlation Plot for numeric variables",
  colors    = c("darkred", "white", "darkblue"),
  outline.color = "gray80",
  tl.col    = "black",
  tl.srt    = 45,
  tl.cex    = 12 / max(1, ncol(corr_matrix) / 14)  # auto-scale text a bit
)

# 3.3 Example scatterplot: Price vs HP (Horsepower)
ggplot(df, aes(x = HP, y = Price)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "ScatterPlot of Price vs Horsepower",
       x = "Horsepower (HP)",
       y = "Price (CA$)")

# ---------------------------------------------------------
# 4. Transform / Encode Variables

# We treat key variables as categorical where appropriate. We convert categorical variables into dummy/indicator variables using `caret::dummyVars()` and create a fully numeric dataset for modeling.
# ---------------------------------------------------------

# 4.1 Treat some variables as factors (categorical)
# Check what Fuel and Colour look like:
table(df$Fuel)
table(df$Color)

df$Fuel <- as.factor(df$Fuel)
df$Color <- as.factor(df$Color)

# 4.2 Ensure binary indicator columns are numeric 0/1
binary_cols <- c("MC", "Auto", "Mfr_G", "ABS", "Abag_1", "Abag_2",
                 "AC", "Comp", "CD", "Clock", "Pwin", "PStr",
                 "Radio", "SpM", "M_Rim", "Tow_Bar")

for (col in binary_cols) {
  if (col %in% names(df)) {
    df[[col]] <- as.numeric(df[[col]])
  }
}

# 4.3 Create dummy variables for categorical features (Fuel, Colour, etc.)
dummy_obj <- dummyVars(" ~ .", data = df)
automobile <- data.frame(predict(dummy_obj, newdata = df))

# Check structure
glimpse(automobile)

# Save cleaned dataset
write_xlsx(automobile, "cleaned_dataset.xlsx")
getwd()

# ---------------------------------------------------------
# 5. Train / Test Split

# We split the encoded dataset into a training set (70%) and a validation set (30%) to evaluate model performance out-of-sample.
# ---------------------------------------------------------

set.seed(123)

# Assume Price is the target
index <- createDataPartition(automobile$Price, p = 0.7, list = FALSE)
train <- automobile[index, ]
valid <- automobile[-index, ]

# Return the number of rows
nrow(train); nrow(valid)

# ---------------------------------------------------------
# 6. Model 1 – Linear Regression

# We fit a multiple linear regression model using all predictors and evaluate performance using RMSE, MAE, and R² on the validation set.
# ---------------------------------------------------------

lm_model <- lm(Price ~ ., data = train)

summary(lm_model)

# Predictions on validation set
pred_lm <- predict(lm_model, newdata = valid)

# Evaluate
RMSE_LM <- RMSE(pred_lm, valid$Price)
MAE_LM  <- MAE(pred_lm,  valid$Price)
R2_LM   <- R2(pred_lm,   valid$Price)

cat("Linear Regression:\n")
cat("  RMSE:", RMSE_LM, "\n")
cat("  MAE :", MAE_LM,  "\n")
cat("  R2  :", R2_LM,   "\n\n")

# Plot the linear regression
plot_lm <- ggplot(data = NULL, aes(x = valid$Price, y = pred_lm)) +
  geom_point(color = "purple", alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, 
              color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Linear Regression for Actual vs Predicted Prices",
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  theme_minimal(base_size = 12)

print(plot_lm)

# ---------------------------------------------------------
# 7. Model 2 – Decision Tree

# We build a regression tree using `rpart`, visualize the tree, and compute the same evaluation metrics.
# ---------------------------------------------------------

tree_model <- rpart(Price ~ ., data = train, method = "anova")

# Plot the decision tree
my_node_lab <- function(x, labs, digits, varlen) {
  paste0(format(round(x$frame$yval, 0), big.mark = ","),
         "\n",
         round(100 * x$frame$n / x$frame$n[1], 1), "%")
}

options(scipen = 999)
rpart.plot(tree_model,
           main = "Decision Tree for Automobile Price",
           node.fun     = my_node_lab,
           fallen.leaves = TRUE,
           box.palette = "Blues",
           shadow.col = "gray",
           nn = TRUE)

# Predictions
pred_tree <- predict(tree_model, newdata = valid)

# Evaluate
RMSE_TREE <- RMSE(pred_tree, valid$Price)
MAE_TREE  <- MAE(pred_tree,  valid$Price)
R2_TREE   <- R2(pred_tree,   valid$Price)

cat("Decision Tree:\n")
cat("  RMSE:", RMSE_TREE, "\n")
cat("  MAE :", MAE_TREE,  "\n")
cat("  R2  :", R2_TREE,   "\n\n")

# ---------------------------------------------------------
# 8. Model 3 – Neural Network

# We construct a feed-forward neural network using `neuralnet`, visualize the architecture, and evaluate its predictive accuracy.
# ---------------------------------------------------------

# 8.1 Build formula (Price as target, all others as predictors)
features_nn <- setdiff(names(train), "Price")
f_nn <- as.formula(paste("Price ~", paste(features_nn, collapse = " + ")))

# 8.2 Train Neural Network (you can play with hidden = c(3,2))
nn_model <- neuralnet(f_nn,
                      data = train,
                      hidden = c(3, 2),
                      linear.output = TRUE)

# Plot the network
par(mar = c(1, 1, 3, 1))   # smaller margins so the plot uses space better

plot(
  nn_model,
  rep          = "best",      # best repetition
  information  = FALSE,       # hide info box
  col.in       = "orange",    # input nodes
  col.hidden   = "blue",      # hidden layer nodes
  col.out      = "green",     # output node
  cex          = 0.6,         # slightly smaller labels
  main         = "Neural Network Architecture for Automobile Price"
)

# 8.3 Predictions
nn_pred_raw <- compute(nn_model, valid[ , features_nn])$net.result
pred_nn <- as.vector(nn_pred_raw)

# Evaluate
RMSE_NN <- RMSE(pred_nn, valid$Price)
MAE_NN  <- MAE(pred_nn,  valid$Price)
R2_NN   <- R2(pred_nn,   valid$Price)

cat("Neural Network:\n")
cat("  RMSE:", RMSE_NN, "\n")
cat("  MAE :", MAE_NN,  "\n")
cat("  R2  :", R2_NN,   "\n\n")

# ---------------------------------------------------------
# 9. Compare Models

# Finally, we compare the three models using RMSE, MAE, and R² on the validation set.
# ---------------------------------------------------------

results <- data.frame(
  Model = c("Linear Regression", "Decision Tree", "Neural Network"),
  RMSE  = c(RMSE_LM, RMSE_TREE, RMSE_NN),
  MAE   = c(MAE_LM,  MAE_TREE,  MAE_NN),
  R2    = c(R2_LM,   R2_TREE,   R2_NN)
)

print(results)

