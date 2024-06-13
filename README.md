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
| `name` | Recipe name |
| `id` | Recipe ID (unique) | 
| `minutes` | Minutes to make recipe | 
| `contributor_id` | User id of user who submitted recipe | 
| `submitted` | Date submitted to website | 
| `tags` | tags | 
| `nutrition` | Nutrition information. Format: calores, total fat, sugar, sodium, protein, saturated fat, carbohydrates. Unit is percentage of daily value | 
| `n_steps` | Number of steps it takes to make the recipe | 
| `steps` | Description of each step | 
| `description` | Recipe description | 
# Data Cleaning and Exploratory Data Analysis

# Assessment of Missingness

# Hypothesis Testing

# Framing a Prediction Problem

# Baseline Model

# Final Model

# Fairness Analysis