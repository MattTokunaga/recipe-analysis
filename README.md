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

This data is honestly already pretty clean, but a few extra steps do need to be taken. First, I merged the two datasets together. This was possible thanks to the unique recipe ID that the website gives each recipe. Then, I averaged the ratings for each recipe to add an average rating column. 

Additionally, the datasets included a weird quirk. Instead of actual lists, some columns, such as "tags," contained information that looked like a list but was actually just one long string. However, splitting the strings and turning them into actual lists was simple using the `.split` command in Python.

I also separated the nutrition information into separate columns. Initially, they were all in the same column and weren't even labeled. Again, splitting these numbers was simple.

This is what our final cleaned data look like:


num_ingredients	1	2	3	4	...	15	16	17	18
n_steps									
1	NaN	27.46	9.80	22.28	...	NaN	245.00	222.50	10.00
2	20.0	28.10	35.88	12.94	...	65.00	87.50	NaN	NaN
3	5.0	113.05	20.33	22.25	...	126.33	151.25	153.12	28.33
...	...	...	...	...	...	...	...	...	...
18	NaN	20.00	133.57	2675.00	...	84.82	75.95	69.20	1834.35
19	NaN	240.00	20222.50	113.18	...	81.71	100.93	313.68	214.50
20	5.0	7581.25	266.67	167.33	...	77.34	118.98	110.51	100.00
20 rows × 18 columns

user_id        False
protein        False
sodium         False
               ...  
name            True
review          True
description     True
Length: 26, dtype: bool
0.0
0.645
0.0
80.30629339398719
{(2, 147.2): 28.499311203582653,
 (2, 245.4): 28.4993039360337,
 (2, 365.4): 28.497246352043746,
 (2, 555.4): 28.499968864307704,
 (3, 147.2): 28.49999459487636,
 (3, 245.4): 28.499986370880052,
 (3, 365.4): 28.49790435788887,
 (3, 555.4): 28.500633347200274,
 (6, 147.2): 28.499927187610297,
 (6, 245.4): 28.499923721724805,
 (6, 365.4): 28.497862235448107,
 (6, 555.4): 28.500608595904634,
 (12, 147.2): 49745056.53594529,
 (12, 245.4): 28.49801499315336,
 (12, 365.4): 28.495963378880013,
 (12, 555.4): 101888724.08148934}
name	id	minutes	contributor_id	...	saturated_fat	carbohydrates	isveg	shuffled
0	vegan parmesan	282837	0	580209	...	13.0	19.0	True	False
1	vegan parmesan	282837	0	580209	...	13.0	19.0	True	False
2	vegan parmesan	282837	0	580209	...	13.0	19.0	True	False
...	...	...	...	...	...	...	...	...	...
232080	bridget s brined turkey	339555	660	769398	...	59.0	0.0	False	False
232081	crock pot baked beans from scratch	322024	660	718022	...	43.0	17.0	False	False
232082	crock pot baked beans from scratch	322024	660	718022	...	43.0	17.0	False	False
232083 rows × 28 columns



25.253542753479408 28.63526687365827
3.3817241201788626
0.0


























3

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                        | nutrition                                                  |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                    |   n_ingredients |   user_id |   recipe_id | date       |   rating | review                                                                                                                                                                                                                                                                                                                                           |   av_rating |   num_ratings |   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbohydrates |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------:|------------:|:-----------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|--------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | ['138.4', '10.0', '50.0', '3.0', '3.0', '19.0', '6.0']     |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat ', 'stirring frequently ', 'until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs ', 'sugar ', 'cocoa powder ', 'vanilla extract ', 'espresso ', 'and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean ', 'about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 |    386585 |      333281 | 2008-11-19 |        4 | These were pretty good, but took forever to bake.  I would send it ended up being almost an hour!  Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut.  They did taste quite rich, though!  Made for My 3 Chefs.                                                                                   |           4 |             1 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               | ['595.1', '46.0', '211.0', '22.0', '13.0', '51.0', '26.0'] |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl ', 'sift together the flours and baking powder', 'set aside', 'in another mixing bowl ', 'blend together the sugars ', 'margarine ', 'and salt until light and fluffy', 'add the eggs ', 'water ', 'and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop ', 'scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !'] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 |    424680 |      453467 | 2012-01-26 |        5 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe! |           5 |             1 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | ['194.8', '20.0', '6.0', '32.0', '22.0', '36.0', '3.0']    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |     29782 |      306168 | 2008-12-31 |        5 | This was one of the best broccoli casseroles that I have ever made.  I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM!                                                                                                                                  |           5 |             4 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
|                                      |        |           |                  |             |                                                                                                                                                                                                                             |                                                            |           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                |                 |           |             |            |          | The photos you took (shapeweaver) inspired me to make this recipe and it actually does look just like them when it comes out of the oven.                                                                                                                                                                                                        |             |               |            |             |         |          |           |                 |                 |
|                                      |        |           |                  |             |                                                                                                                                                                                                                             |                                                            |           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                |                 |           |             |            |          | Thanks so much for sharing your recipe shapeweaver. It was wonderful!  Going into my family's favorite Zaar cookbook :)                                                                                                                                                                                                                          |             |               |            |             |         |          |           |                 |                 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | ['194.8', '20.0', '6.0', '32.0', '22.0', '36.0', '3.0']    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |   1196280 |      306168 | 2009-04-13 |        5 | I made this for my son's first birthday party this weekend. Our guests INHALED it! Everyone kept saying how delicious it was. I was I could have gotten to try it.                                                                                                                                                                               |           5 |             4 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | ['194.8', '20.0', '6.0', '32.0', '22.0', '36.0', '3.0']    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray ', 'set aside', 'in a large bowl mix together broccoli ', 'soup ', 'one cup of cheese ', 'garlic powder ', 'pepper ', 'salt ', 'milk ', '1 cup of french onions ', 'and soy sauce', 'pour into baking dish ', 'sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly ', 'about 10 more minutes']                                                                                                                                                                                                                                                                                                                      | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |    768828 |      306168 | 2013-08-02 |        5 | Loved this.  Be sure to completely thaw the broccoli.  I didn&#039;t and it didn&#039;t get done in time specified.  Just cooked it a little longer though and it was perfect.  Thanks Chef.                                                                                                                                                     |           5 |             4 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |



# Assessment of Missingness

# Hypothesis Testing

# Framing a Prediction Problem

# Baseline Model

# Final Model

# Fairness Analysis