# Regression Based Recipe Analysis and Prediction
# Introduction
Hello! The purpose of this project is to do some analysis on recipes, as well as do a little bit of machine learning. This project is primarily split into two parts: the first part is conducting a simple permutation test, and the second part is incorporating machine learning to predict how long a recipe will take. The permutation test answers the question: if a recipe has the word "easy" in the title, is it more likely to have ingredients with longer names? 

## The data

First let's start with the data. What exactly are we looking at? Well we actually have two separate datasets, both from the same place: food.com. The first dataset contains information about 83782 different recipes from their website. Here's what the first few entries look like:

| name                                 |     id |   minutes |   n_steps |
|:-------------------------------------|-------:|----------:|----------:|
| 1 brownies in the world    best ever | 333281 |        40 |        10 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        12 |
| 412 broccoli casserole               | 306168 |        40 |         6 |
| millionaire pound cake               | 286009 |       120 |         7 |
| 2000 meatloaf                        | 475785 |        90 |        17 |

In reality though, there are way more columns than this. I just chose to show the ones that would look the cleanest, but here's a description of all the columns:

| Column name | Description |
| `name`      | Recipe name |
| `id`        | Recipe ID (unique) | 
| `minutes`   | Minutes to make recipe | 
| `tags`      | tags | 
| `nutrition` | Nutrition information. Format: calores, total fat, sugar, sodium, protein, saturated fat, carbohydrates. Unit is percentage of daily value   | 
| `n_steps`   | Number of steps it takes to make the recipe | 

Additionally, there's actually another dataset, although it contains mostly the same information. The only difference is that it contains reviews and ratings for the recipes. 

|          user_id |   recipe_id |   rating |
|-----------------:|------------:|---------:|
|      1.29371e+06 |       40893 |        5 |
| 126440           |       85009 |        5 |
|  57222           |       85009 |        5 |
| 124416           |      120345 |        0 |
|      2.00019e+09 |      120345 |        2 |

The columns in this one are pretty self explanatory. The only thing I didn't show here was the column that actually contains the text of the review. However, I didn't use that in my analysis.

# Data Cleaning and Exploratory Data Analysis

## Data Cleaning
This data is honestly already pretty clean, but a few extra steps do need to be taken. First, I merged the two datasets together. This was possible thanks to the unique recipe ID that the website gives each recipe. Then, I averaged the ratings for each recipe to add an average rating column. 

Additionally, the datasets included a weird quirk. Instead of actual lists, some columns, such as "tags," contained information that looked like a list but was actually just one long string. However, splitting the strings and turning them into actual lists was simple using the `.split` command in Python.

I also separated the nutrition information into separate columns. Initially, they were all in the same column and weren't even labeled. Again, splitting these numbers was simple.

This is the nutrition information in our final cleaned data looks like (showing every column would be extremely wide and doesn't work):


|   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbohydrates |
|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|
|      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |
|      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |
|      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
|      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
|      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |

## Exploratory Data Analysis

One variable I decided to look at was the number of ingredients in each recipe. I was interested in the distribution of the number of ingredients, and decided to make a plot.
<iframe
  src="assets/ingredient_amounts.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The graph clearly shows that most recipes have around 8 ingredients, with anything above 20 or so being extremely unlikely. However, there is still a positive skew, mostly due to the fact that you can't have negative outliers in this context. You also can't really have exactly 0 ingredients for obvious reasons. 

# Assessment of Missingness

# Hypothesis Testing

# Framing a Prediction Problem

# Baseline Model

# Final Model

# Fairness Analysis