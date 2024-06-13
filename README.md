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
This data is honestly already pretty clean, but a few extra steps do need to be taken. First, I merged the two datasets together. This was possible thanks to the unique recipe ID that the website gives each recipe. Then, I averaged the ratings for each recipe to add an average rating column. This was after filling any missing ratings with 0.

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

I also decided to look at the relationship between two variables, a process known as **bivariate analysis**. 

However, this wasn't as fruitful. Most of the variables I looked at didn't seem to be related in obvious ways. For example, here's a plot of the relationship between the number of ingredients and the minutes the recipe takes:
<iframe
  src="assets/minutes_ingredients.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

First, I need to point out that I had to manually set a cutoff for the recipes that would be shown. The number of minutes in each recipe had some very extreme outliers, and so I limited the graph to only show recipes that take less than about 1500 minutes. 

There isn't exactly a strong trendline. The recipes with over 20 ingredients maybe seem less likely to take a long time, but this could just be due to a small sample size. The overwhelming majority of recipes use between 0 and 20 ingredients and take less than 100 minutes. I think this simple fact accounts for the cluster that seems to appear, and not any actual relationship between the variables.

Here's an interesting pivot table:
|   n_steps |   1 |        2 |         3 |       4 |       5 |
|----------:|----:|---------:|----------:|--------:|--------:|
|         1 | nan |  27.459  |   9.80132 | 22.2837 | 24.1061 |
|         2 |  20 |  28.1034 |  35.8803  | 12.9437 | 20.8824 |
|         3 |   5 | 113.053  |  20.3306  | 22.2534 | 44.4728 |
|         4 |   5 | 486.832  | 139.039   | 51.9708 | 36.4664 |
|         5 |   5 | 348.797  |  47.9932  | 91.3391 | 74.3867 |

This is how you read this table: If you take all the recipes with, for example, 3 steps and 2 ingredients, and take the average of the minutes for those recipes, you get 113.053. The rows represent the number of steps, and the columns the number of ingredients. From this limited preview, it actually seems as if the recipes with 2 ingredients only tended to take a long time. However, any present trends aren't exactly clear.

# Assessment of Missingness

This dataset doesn't have much missing data, but there is some missing. At the very beginning I filled any missing ratings with 0. The only other columns that contained any missing data at all are the "name" column, the "review" column, and the "description" column. However, I would have to say that I don't think any of these would be considered NMAR, or "Not Missing at Random." The reason is that I believe these are missing by design. If there is no description listed, for example, it is probably because the person that submitted the recipe didn't put a description. That would be missing by design, rather than not missing at random.



# Hypothesis Testing

# Framing a Prediction Problem

# Baseline Model

# Final Model

# Fairness Analysis