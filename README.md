# Regression Based Recipe Analysis and Prediction
# Introduction
Hello! The purpose of this project is to do some analysis on recipes, as well as do a little bit of machine learning. This project is primarily split into two parts: the first part is conducting a simple permutation test, and the second part is incorporating machine learning to predict how long a recipe will take. The permutation test answers the question: if a recipe has the word "easy" in the title, is it more likely to have ingredients with longer names? 

## The data

First let's start with the data. What exactly are we looking at? Well we actually have two separate datasets, both from the same place: food.com. The first dataset contains information about 83782 different recipes from their website. Here's what the first few entries look like:


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

| name                                 |     id |   minutes | tags                                                                                                                                                                                                                                                                                               |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|:-------------------------------------|-------:|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 brownies in the world    best ever | 333281 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat , stirring frequently , until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs , sugar , cocoa powder , vanilla extract , espresso , and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean , about 25 to 30 minutes', 'remove from the oven and cool completely before cutting']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 1 in canada chocolate chip cookies   | 453467 |        45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy', 'add the eggs , water , and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop , scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 412 broccoli casserole               | 306168 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| millionaire pound cake               | 286009 |       120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['freheat the oven to 300 degrees', 'grease a 10-inch tube pan with butter , dust the bottom and sides with flour , and set aside', 'in a large mixing bowl , cream the butter and sugar with an electric mixer and add the eggs one at a time , beating after each addition', 'alternately add the flour and milk , stirring till the batter is smooth', 'add the two extracts and stir till well blended', 'scrape the batter into the prepared pan and bake till a cake tester or knife blade inserted in the center comes out clean , about 1 1 / 2 hours', 'cool the cake in the pan on a rack for 5 minutes , then turn it out on the rack to cool completely']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2000 meatloaf                        | 475785 |        90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['pan fry bacon , and set aside on a paper towel to absorb excess grease', 'mince yellow onion , red bell pepper , and add to your mixing bowl', 'chop garlic and set aside', 'put 1tbsp olive oil into a saut pan , along with chopped garlic , teaspoons white pepper and a pinch of kosher salt', 'bring to a medium heat to sweat your garlic', 'preheat oven to 350f', 'coarsely chop your baby spinach add to your heated pan , stir frequently for approximately 5 min to wilt', 'add your spinach to the mixing bowl', 'chop your now cooled bacon , and add it to the mixing bowl', 'add your meatloaf mix to the bowl , with one egg and mix till thoroughly combined', 'add your goat cheese , one egg , 1 / 8 tsp white pepper and 1 / 8 tsp of kosher salt and mix till thoroughly combined', 'transfer to a 9x5 meatloaf pan , and cook for 60 min or until the internal temperature is at least 160f', 'let stand for 5min', 'melt 1tbsp unsalted butter into a frying pan , and cook up to three eggs at a time', 'crack each egg into a separate dish , in order to prevent egg shells from reaching the pan , then add salt and pepper to taste', 'wait until the egg whites are firm looking , but slightly runny on top before flipping your eggs', 'after flipping , wait 10~20 seconds before removing each egg and placing it over your slices of meatloaf'] |





# Data Cleaning and Exploratory Data Analysis

# Assessment of Missingness

# Hypothesis Testing

# Framing a Prediction Problem

# Baseline Model

# Final Model

# Fairness Analysis