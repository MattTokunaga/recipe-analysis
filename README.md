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

In order to determine the missigness mechanism of the "review" column, I decided to run permutation tests with the "review" column and the "n_steps" and "minutes" columns. I wanted to see if recipes with missing reviews tended to have a higher number of steps or minutes. Here is the result:

<iframe
  src="assets/permgraph.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The red line represents the observed test statistic, and the blue bars represent obtained experimental statistics. As you can see, the difference between the mean number of steps for recipes with and without missing reviews was abnormally high, higher than any experimental result. Thus we can conclude that they are in fact  related, so reviews are missing at random with respect to the number of steps.

# Hypothesis Testing

The null hypothesis for my permutation test will be that recipes with "easy" in the title have, on average, the same ingredient name length than recipes without "easy" in the title. The alternative hypothesis is that recipes with "easy" in the title tend to have longer ingredient names. I figured that this could plausibly be the case, since if a recipe is "easy" it might be that way because it uses preprocessed foods with longer names.

My choice of test statistic was: average average ingredient name length for easy recipes - average average ingredient name length of non-easy recipes.

This is a bit confusing, so I'll explain. First, for each recipe you look at all of its ingredients, and you average the lengths. Then, for the recipes with easy, you average all of those averages. Same for non-easy recipes. Then you subtract them, and if the alternative hypothesis is true you should get a notably positive number. I used a significance level of 0.05, which is standard. However, it didn't really matter, since the results of my permutation test gave me a p-value of exactly 0. 

This result means we would reject the null hypothesis. We can't prove the alternative hypothesis, but it does seem to be that there is a relationship between a recipe being easy or not and how long the ingredient names are.

<iframe
  src="assets/sect4graph.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Here, like last time, you can see that our observed statistic isn't even close to any of the experimental test statistics.

# Framing a Prediction Problem

Using machine learning, I set out to use features in the data to predict how long a recipe would take, in minutes. This is a regression problem, since we are trying to predict a continuous, quantitative variable. I chose this variable to predict because, out of all of them, it seems like it would be the most related to other variables. I'm evaluating my model using root mean square error, which is a good metric for continuous data. I thought it would be better to use this than something like R squared because I wanted to know, generally, how far off my predictions were, and not how correlated they were.

# Baseline Model

I have to come clean. I actually simplified the problem for myself. I decided to only keep the smallest 99% of recipes with respect to the minutes they take. This is because I figured having the huge outliers, like the recipes that take over one million minutes, would make it impossible to build a model that scores well. Not only that, but needing to account for those outliers wouls probably shift the model parameters enough that it wouldn't even perform well for the most common recipes, the ones that take a reasonable amount of time. If you believe that makes me a cheater and a liar, I humbly accept those titles.

To start with, I set up a very simple model. Using the `sklearn` library for Python, I set up a simple linear regression model that takes in two variables, the number of steps and the average rating, and tries to predict the minutes it takes to prepare the recipe. These two features are both quantitative, so no encoding was necessary. I also didn't transform them in any way.

The performance was, as excpected, not great. The root mean square error was around 80, which isn't very good if you're trying to predict the time for recipes that take usually around 60 minutes. However, for only using two very basic features, this was promising. 

Here's the first few predictions, as well as the real values:

|   Real Minutes |   Predicted Minutes |   Difference |
|---------------:|--------------------:|-------------:|
|            375 |             90.9077 |    284.092   |
|              5 |             42.5247 |    -37.5247  |
|             40 |             44.511  |     -4.51097 |
|              5 |             38.291  |    -33.291   |
|             45 |             64.6864 |    -19.6864  |


# Final Model

In order to improve my model, I simply added more columns. I included information from the "tags," "name," "num_ratings," "calories," "n_steps," "av_rating," and "n_ingredients" columns. However, I only used those last 3 in their raw forms. For the calories and number of ratings, I turned them into binary variables. The way this works is you pick a threshold and say anything above that is a 1 and anything below that is a 0. 

I "binarized" these two columns in particular because it felt plausible that a recipe with more calories might be simply more food, or at least require more processing, and thus having an indicator for high caloric recipes would be useful. For the number of ratings, I thought that maybe recipes with more ratings would be more "ordinary," in that they probably wouldn't be massive outliers in any particular area. I figured this because an abnormal recipe would be less likely to actually get made and rated.

For the "tags" and "name" columns, I had to get a bit more creative. That's because these columns aren't quantitative, meaning they aren't numbers. That means they can't be used in their raw forms, and so I had to find a way to transform them.

For the "tags" feature, I decided to use one-hot-encoding. This process involves making a new column for every single possible tag, and then putting a 1 if that particular recipe has that tag and a 0 otherwise. Obviously this increases the complexity by quite a bit, and so it took a while to train. However, I had good reason to believe this would help. Certain tags, something like "10 minutes or less," would be highly related to the time the recipe is meant to take. 

I decided to do a similar thing with the "name" column. My general idea was that seeing which adjectives were in the title could give good indications on recipe duration. For example, if a recipe had "easy" in the title, or if it had "gourmet" in the title, it might be faster or slower than average, respectively. So first I found a big list of adjectives [here](https://patternbasedwriting.com/elementary_writing_success/list-4800-adjectives/). Then I made a list of every single word that appears even once in any recipe name. After that, I found the ones that are on the adjective list, meaning they're adjectives, and filtered for ones that are used over 5000 times total. Finally, I used the same one-hot-encoding trick for these adjectives.

For my modeling algorithm, I chose a simple `sklearn` Linear Regression model. I played around with a couple other regressino options, but none seemed to outperform this one, and this one is the most basic. 

My main engineered feature, the one hot encoded columns, didn't really require any hyperparameter optimization. However, one thing I could optimize was the thresholds for the binarizers. I decided to, for each column, set the threshold to be the quintiles (20th, 40th, 60th, 80th percentiles. 0 and 100 wouldn't be very helpful) for their respective variables. This meant, in total, I had 16 hyperparameter combinations to check, since there were 4 options for each binarizer.

Unfortunately, this hyperparameter optimization didn't really matter. Most of the combinations were basically the exact same, and I suspect this is because the binary columns didn't really play much of a role in the calculations anyways.

The performance of this final model is much, much better than the initial baseline model. The baseline model had an RMSE of about 80, while the final model had an RMSE of around 28. 

# Fairness Analysis

To do fairness analysis, I decided to split my data into two groups: vegetarian recipes and non-vegetarian recipes. To do this, I added a boolean column indicating whether or not the name of the recipe included the word "vegan" or "vegetarian." I evaluated both using the RMSE metric and did a permutation test. My null hypothesis was that my final model works equally well on recipes with either "vegan" or "vegetarian" in the title and recipes without those words in the title. My alternative hypothesis was that the model was significantly better on one group than the other. Importantly, I did not specify which group it performed better for, so this is a two tailed hypothesis. I picked my test statistic accordingly, and went for an absolute difference between the RMSE of the model for both groups. 

The observed statistic was about 3.4, meaning that the RMSE was 3.4 higher for one group than the other. The information about which group it was isn't built into the test statistic, but a simple `print` statement reveals that for vegetarian recipes the RMSE was about 25, while it was about 28.5 for all other recipes. Thus our model actually performed better on vegetarian recipes. 

However, was this statistically significant? Using a significance level of 0.05, I ran a permutation test, shuffling the vegetarian indicator column, and then finding the absolute difference in RMSE for both columns I only did 100 shuffles since each one took about 3 seconds, which really adds up. I don't think it mattered too much though, since I got a p value of exactly 0. This means that in 0 out of the 100 trials did we get a test statistic as or more extreme than the observed statistic. This means we must reject the null hypothesis. This is just conjecture, but I think it would make sense if the model is just better for non-outliers, and the sample size of vegetarian recipes might be small enough that there are no outliers.

Thank you for reading my analysis!