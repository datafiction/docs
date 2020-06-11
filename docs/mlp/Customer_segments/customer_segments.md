
# Machine Learning Engineer Nanodegree
## Unsupervised Learning
## Project: Creating Customer Segments

Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

## Getting Started

In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.

The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.

Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
# Import supplementary visualizations code visuals.py
import visuals as vs
import seaborn as sns

# Pretty display for notebooks
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?") 
```

    Wholesale customers dataset has 440 samples with 6 features each.


-----------

## Data Exploration
In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.

Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

*** Lets see what are the column names and their types***


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 440 entries, 0 to 439
    Data columns (total 6 columns):
    Fresh               440 non-null int64
    Milk                440 non-null int64
    Grocery             440 non-null int64
    Frozen              440 non-null int64
    Detergents_Paper    440 non-null int64
    Delicatessen        440 non-null int64
    dtypes: int64(6)
    memory usage: 20.7 KB


*** Head part of the data ***


```python
data.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12669</td>
      <td>9656</td>
      <td>7561</td>
      <td>214</td>
      <td>2674</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7057</td>
      <td>9810</td>
      <td>9568</td>
      <td>1762</td>
      <td>3293</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6353</td>
      <td>8808</td>
      <td>7684</td>
      <td>2405</td>
      <td>3516</td>
      <td>7844</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13265</td>
      <td>1196</td>
      <td>4221</td>
      <td>6404</td>
      <td>507</td>
      <td>1788</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22615</td>
      <td>5410</td>
      <td>7198</td>
      <td>3915</td>
      <td>1777</td>
      <td>5185</td>
    </tr>
  </tbody>
</table>
</div>



##### Lets look how each feature propagates from customer to customers.


```python
from matplotlib import cm
test_data = data.loc[0:200]
test_data.plot(colormap=cm.cubehelix,figsize=(16, 12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10de94950>




![png](output_11_1.png)


Fresh and grocery are really high demanding items! But this is not the global nature of the data. Detail will be discussed later.

#### Lets look their propertion on small section of the data


```python
test_data = data.loc[100:150]
test_data.plot.barh(stacked=True,figsize=(16, 15) );
```


![png](output_14_0.png)


Customer at index 125 is really heavy consumer of fresh!

***Little bit of statistics of each columns:***


```python
# Display a description of the dataset
display(data.describe())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>


In general the mean for each feature seems much higher than the median(50%-Quartile), so this data seems somewhat **skewed**. We can visualize them as well in the histogram comming in next blocks.

*** Histogram for individual data***


```python
plt.figure(figsize = (14,14))
bins = [100, 100, 100, 100, 100, 100]
colors = sns.color_palette("bright", 6) 
items = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']
ks = [1,2,3,4,5,6]

# plot individual histograms
for color,item,k in zip(colors,items,ks):
    plt.subplot(3,2,k)
    plt.title("plot of " + item)
    sns.distplot(data[item],bins = 50, kde=True, color=color)

```


![png](output_20_0.png)


Most of these features have peak near zero to 1000. **Grocery** is pretty much flat among them. We can cut a section from each of these plots and compare between them as shown below:

*** Putting all together***


```python
# Plot all together
plt.figure(figsize = (14,6))
plt.title("distribution of all items")
for ibin,color,item in zip(bins,colors,items):
    sns.distplot(data[item],bins = ibin, kde=True, color=color)
plt.legend([item for item in items])
```




    <matplotlib.legend.Legend at 0x1051439d0>




![png](output_23_1.png)


Peak of the distribution shifts to the right while moving from feature Lets make it much lear as shown below:

*** Lets visualize most dynamic part of the data***


```python
# view most dynamic part of the data
plt.figure(figsize=(14,10))
plt.title("Viewing most dynamic part of the data")
for ibin,color,item in zip(bins,colors,items):
    sns.distplot(data[data[item]<= 5000][item],bins = ibin, kde=True, color=color)
plt.legend([item for item in items])
```




    <matplotlib.legend.Legend at 0x111c56050>




![png](output_26_1.png)


Wow! Look at this trend : **Detergent -> Delicatessen-> Fresh-I -> Frozen -> Milk -> Grocery -> Fresh-II** . Fresh has two peak representing two types of purcheser : household(low price) and might be resturents(high price) etc. Data skewed to the right as pridected before.

--------

### Implementation: Selecting Samples
To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.


```python
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [12, 125, 412]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset:")

display(samples)
samples.plot.barh(stacked=True,figsize=(10,2) );

# show devations of each feature from its mean measured by std
mean_data = np.mean(data)
std_data = np.std(data)
deviation_samples = (samples - mean_data) / std_data


print ("\nDeviation of chosen samples of wholesale customers dataset in mean+deviation*std:")
display(deviation_samples)


```

    Chosen samples of wholesale customers dataset:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31714</td>
      <td>12319</td>
      <td>11757</td>
      <td>287</td>
      <td>3881</td>
      <td>2931</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76237</td>
      <td>3473</td>
      <td>7102</td>
      <td>16538</td>
      <td>778</td>
      <td>918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>3605</td>
      <td>12400</td>
      <td>98</td>
      <td>2970</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>


    
    Deviation of chosen samples of wholesale customers dataset in mean+deviation*std:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.560499</td>
      <td>0.884800</td>
      <td>0.400925</td>
      <td>-0.574313</td>
      <td>0.209873</td>
      <td>0.499176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.084854</td>
      <td>-0.315148</td>
      <td>-0.089470</td>
      <td>2.776994</td>
      <td>-0.441685</td>
      <td>-0.215439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.942242</td>
      <td>-0.297242</td>
      <td>0.468664</td>
      <td>-0.613289</td>
      <td>0.018584</td>
      <td>-0.519319</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_30_4.png)


#### Quartile Visualization of sample


```python
plt.figure(figsize = (12,5))
percentiles = data.rank(pct=True)
percentiles = 100*percentiles.round(decimals=3)
percentiles = percentiles.iloc[indices]
display(percentiles)
print "Quartile Visualization"
sns.heatmap(percentiles, vmin=1, vmax=99, annot=True)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>93.6</td>
      <td>90.2</td>
      <td>79.1</td>
      <td>10.7</td>
      <td>74.5</td>
      <td>89.8</td>
    </tr>
    <tr>
      <th>125</th>
      <td>99.8</td>
      <td>47.7</td>
      <td>63.4</td>
      <td>98.2</td>
      <td>48.6</td>
      <td>48.6</td>
    </tr>
    <tr>
      <th>412</th>
      <td>1.8</td>
      <td>49.3</td>
      <td>80.9</td>
      <td>3.2</td>
      <td>69.3</td>
      <td>4.8</td>
    </tr>
  </tbody>
</table>
</div>


    Quartile Visualization





    <matplotlib.axes._subplots.AxesSubplot at 0x113465150>




![png](output_32_3.png)


--------------

### Question 1
Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
**What kind of establishment (customer) could each of the three samples you've chosen represent?**  
**Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

**Answer:**

There are three information hiden in samples which are
(1) Total values: It helps to know typr of wholesale distributer
(2) Mean value of each category and deviations:  It helps to know types of distributer

Lets go sample by sample:

**Sample-1:** This customer a heavy consumer of Fresh items, milk and Delicatessen. Spending on Grocery is also above 75th quartile. Frozen is pretty much less and below 25th quartile and mean. It should be **Supermarket**, because all kinds of produces are sold much more than average value except the frozen.

**Sample-2:** This customer is crazy at Fresh. It touched 100th quartile.Second most purchased items are frozens. Grocery is little below the mean value but pretty above the 50th quartile value. Rest of them are below 5oth quartile.This should be **Greengrocery**, be cause the main produce being sold is fresh produce and frozen  while the total amount of produces is below average

**Sample-3:** For this customer, first priority is Grocery which is above 75th quartile range and pretty above mean. Next priority os Detergent paper which is above 50th quartile range and also above the mean. Milk is below mean and median. Rest of the other are blow mean and 25th quartile range. It is **Convenience store**, because the main produce being sold is grocery, while the total amount of produces is below average



----------

### Implementation: Feature Relevance
One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.

In the code block below, you will need to implement the following:
 - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
 - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
   - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
 - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
 - Report the prediction score of the testing set using the regressor's `score` function.


```python
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy()
targets = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

# train and evaluate for each feature
scores = {}
for target_feature in targets:
    label = new_data[target_feature]
    left_features = new_data.drop([target_feature], axis=1)

    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(left_features, label, test_size=0.25, random_state=1)

    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=1)
    regressor.fit(X_train, y_train)

    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    scores[target_feature]= score

# display features and scores 
result = pd.DataFrame(scores, index=['Score'])
display(result)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Delicatessen</th>
      <th>Detergents_Paper</th>
      <th>Fresh</th>
      <th>Frozen</th>
      <th>Grocery</th>
      <th>Milk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Score</th>
      <td>-0.429125</td>
      <td>0.815241</td>
      <td>-0.923374</td>
      <td>-0.649574</td>
      <td>0.795768</td>
      <td>0.51585</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize = (12,12))
result.plot(kind = "bar",figsize=(8,8))
plt.axhline(0, color='k')
```




    <matplotlib.lines.Line2D at 0x113bc6350>




    <matplotlib.figure.Figure at 0x113bc6550>



![png](output_39_2.png)


----------

### Question 2
*Which feature did you attempt to predict? What was the reported prediction score? Is this feature is necessary for identifying customers' spending habits?*  
**Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

**Answer:**

I am going to predict `Detergents_Paper` or `Grocery`. Reported prediction score for them are 0.815241 and 0.795768 respectively. No! this feature is less important for identifying customer's spending habits. The reason is following:

1. A predication with higher score (close to 1.0) implies that the training features are likely to predicate the label feature. Alternalely, label feature is more dependent on the other features, which makes it unnecessary for identifying customer's spending habits.

2. Predication with lower score (negative value) indicates the labeling feature is independent to others and necessary for learning algorithm.



-----------

### Visualize Feature Distributions
To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.


```python
plt.figure(figsize = (16,21))
sns.pairplot(data)
```




    <seaborn.axisgrid.PairGrid at 0x113bc6c90>




    <matplotlib.figure.Figure at 0x113bc6b90>



![png](output_45_2.png)


#### Study of variance covariance of the data


```python
data.cov()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.599549e+08</td>
      <td>9.381789e+06</td>
      <td>-1.424713e+06</td>
      <td>2.123665e+07</td>
      <td>-6.147826e+06</td>
      <td>8.727310e+06</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>9.381789e+06</td>
      <td>5.446997e+07</td>
      <td>5.108319e+07</td>
      <td>4.442612e+06</td>
      <td>2.328834e+07</td>
      <td>8.457925e+06</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-1.424713e+06</td>
      <td>5.108319e+07</td>
      <td>9.031010e+07</td>
      <td>-1.854282e+06</td>
      <td>4.189519e+07</td>
      <td>5.507291e+06</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>2.123665e+07</td>
      <td>4.442612e+06</td>
      <td>-1.854282e+06</td>
      <td>2.356785e+07</td>
      <td>-3.044325e+06</td>
      <td>5.352342e+06</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-6.147826e+06</td>
      <td>2.328834e+07</td>
      <td>4.189519e+07</td>
      <td>-3.044325e+06</td>
      <td>2.273244e+07</td>
      <td>9.316807e+05</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>8.727310e+06</td>
      <td>8.457925e+06</td>
      <td>5.507291e+06</td>
      <td>5.352342e+06</td>
      <td>9.316807e+05</td>
      <td>7.952997e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (10,10))
sns.heatmap(data.cov(),annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x115490890>




![png](output_48_1.png)


This plot of covariance gives us little bit more idea about possible correlation between variables. This is symmetric plot. Diagonal values represents the variance where as non diagonal values gives some information about correlation. For example we can look **Detergents_Paper** and **Grocery**. Lets look them in separate plot below:

#### plot of Detergents_Paper with Grocery


```python
plt.figure(figsize = (8,8))
sns.jointplot(x="Detergents_Paper", y="Grocery", data=data,size=15,kind = 'reg');
```


    <matplotlib.figure.Figure at 0x115ec98d0>



![png](output_51_1.png)


--------

### Question 3
*Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
**Hint:** Is the data normally distributed? Where do most of the data points lie? 

**Answer:**

1. **Are there any pairs of features which exhibit some degree of correlation?**
    
   Yes! `Detergents_Paper` and `Grocery` are highly correlated. There may be little correlation between `Detergents_Paper` and `Milk`, `Grocery` and `Milk` as well.
    
2. **Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict?**

    Yes! It confirms my suspicions about the relevance of the feature in Question 2.
    
3. **How is the data for those features distributed?**

    The data distribution plot looks more like a F distribution.


----------

## Data Preprocessing
In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

### Implementation: Feature Scaling
If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.

In the code block below, you will need to implement the following:
 - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
 - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.


```python
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
sns.pairplot(log_data)
```




    <seaborn.axisgrid.PairGrid at 0x110d1cbd0>




![png](output_58_1.png)


### Observation
After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).

Run the code below to see how the sample data has changed after having the natural logarithm applied to it.


```python
# Display the log-transformed sample data
display(log_samples)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.364514</td>
      <td>9.418898</td>
      <td>9.372204</td>
      <td>5.659482</td>
      <td>8.263848</td>
      <td>7.983099</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.241602</td>
      <td>8.152774</td>
      <td>8.868132</td>
      <td>9.713416</td>
      <td>6.656727</td>
      <td>6.822197</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Outlier Detection
Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.

In the code block below, you will need to implement the following:
 - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
 - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
 - Assign the calculation of an outlier step for the given feature to `step`.
 - Optionally remove data points from the dataset by adding indices to the `outliers` list.

**NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
Once you have performed this implementation, the dataset will be stored in the variable `good_data`.


```python
log_data['sn'] = log_data.index
bad_indexes = {}

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    feature_data = log_data[feature]
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(feature_data, 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(feature_data, 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    feature_outliers = log_data[~((feature_data >= Q1 - step) & (feature_data <= Q3 + step))]
    
    
    # Display the outliers
    print ("Data points considered outliers for the feature '{}':".format(feature))
    display(feature_outliers)
    
    
    
    for i, r in feature_data.iteritems():
        if not (Q1 - step <= r <= Q3 + step):
            if i not in bad_indexes:
                bad_indexes[i] = 1
            else:
                bad_indexes[i] += 1
    
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [i for i, n in bad_indexes.items() if n > 1]



# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
outliars_data = log_data.iloc[log_data.index[outliers],:]
```

    Data points considered outliers for the feature 'Fresh':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
      <td>65</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
      <td>66</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.389072</td>
      <td>9.163249</td>
      <td>9.575192</td>
      <td>5.645447</td>
      <td>8.964184</td>
      <td>5.049856</td>
      <td>81</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.098612</td>
      <td>7.979339</td>
      <td>8.740657</td>
      <td>6.086775</td>
      <td>5.407172</td>
      <td>6.563856</td>
      <td>95</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.135494</td>
      <td>7.869402</td>
      <td>9.001839</td>
      <td>4.976734</td>
      <td>8.262043</td>
      <td>5.379897</td>
      <td>96</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
      <td>128</td>
    </tr>
    <tr>
      <th>171</th>
      <td>5.298317</td>
      <td>10.160530</td>
      <td>9.894245</td>
      <td>6.478510</td>
      <td>9.079434</td>
      <td>8.740337</td>
      <td>171</td>
    </tr>
    <tr>
      <th>193</th>
      <td>5.192957</td>
      <td>8.156223</td>
      <td>9.917982</td>
      <td>6.865891</td>
      <td>8.633731</td>
      <td>6.501290</td>
      <td>193</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2.890372</td>
      <td>8.923191</td>
      <td>9.629380</td>
      <td>7.158514</td>
      <td>8.475746</td>
      <td>8.759669</td>
      <td>218</td>
    </tr>
    <tr>
      <th>304</th>
      <td>5.081404</td>
      <td>8.917311</td>
      <td>10.117510</td>
      <td>6.424869</td>
      <td>9.374413</td>
      <td>7.787382</td>
      <td>304</td>
    </tr>
    <tr>
      <th>305</th>
      <td>5.493061</td>
      <td>9.468001</td>
      <td>9.088399</td>
      <td>6.683361</td>
      <td>8.271037</td>
      <td>5.351858</td>
      <td>305</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.098612</td>
      <td>5.808142</td>
      <td>8.856661</td>
      <td>9.655090</td>
      <td>2.708050</td>
      <td>6.309918</td>
      <td>338</td>
    </tr>
    <tr>
      <th>353</th>
      <td>4.762174</td>
      <td>8.742574</td>
      <td>9.961898</td>
      <td>5.429346</td>
      <td>9.069007</td>
      <td>7.013016</td>
      <td>353</td>
    </tr>
    <tr>
      <th>355</th>
      <td>5.247024</td>
      <td>6.588926</td>
      <td>7.606885</td>
      <td>5.501258</td>
      <td>5.214936</td>
      <td>4.844187</td>
      <td>355</td>
    </tr>
    <tr>
      <th>357</th>
      <td>3.610918</td>
      <td>7.150701</td>
      <td>10.011086</td>
      <td>4.919981</td>
      <td>8.816853</td>
      <td>4.700480</td>
      <td>357</td>
    </tr>
    <tr>
      <th>412</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
      <td>412</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Milk':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
      <td>86</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.220590</td>
      <td>4.718499</td>
      <td>6.656727</td>
      <td>6.796824</td>
      <td>4.025352</td>
      <td>4.882802</td>
      <td>98</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
      <td>154</td>
    </tr>
    <tr>
      <th>356</th>
      <td>10.029503</td>
      <td>4.897840</td>
      <td>5.384495</td>
      <td>8.057377</td>
      <td>2.197225</td>
      <td>6.306275</td>
      <td>356</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Grocery':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
      <td>75</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
      <td>154</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Frozen':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>8.431853</td>
      <td>9.663261</td>
      <td>9.723703</td>
      <td>3.496508</td>
      <td>8.847360</td>
      <td>6.070738</td>
      <td>38</td>
    </tr>
    <tr>
      <th>57</th>
      <td>8.597297</td>
      <td>9.203618</td>
      <td>9.257892</td>
      <td>3.637586</td>
      <td>8.932213</td>
      <td>7.156177</td>
      <td>57</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
      <td>65</td>
    </tr>
    <tr>
      <th>145</th>
      <td>10.000569</td>
      <td>9.034080</td>
      <td>10.457143</td>
      <td>3.737670</td>
      <td>9.440738</td>
      <td>8.396155</td>
      <td>145</td>
    </tr>
    <tr>
      <th>175</th>
      <td>7.759187</td>
      <td>8.967632</td>
      <td>9.382106</td>
      <td>3.951244</td>
      <td>8.341887</td>
      <td>7.436617</td>
      <td>175</td>
    </tr>
    <tr>
      <th>264</th>
      <td>6.978214</td>
      <td>9.177714</td>
      <td>9.645041</td>
      <td>4.110874</td>
      <td>8.696176</td>
      <td>7.142827</td>
      <td>264</td>
    </tr>
    <tr>
      <th>325</th>
      <td>10.395650</td>
      <td>9.728181</td>
      <td>9.519735</td>
      <td>11.016479</td>
      <td>7.148346</td>
      <td>8.632128</td>
      <td>325</td>
    </tr>
    <tr>
      <th>420</th>
      <td>8.402007</td>
      <td>8.569026</td>
      <td>9.490015</td>
      <td>3.218876</td>
      <td>8.827321</td>
      <td>7.239215</td>
      <td>420</td>
    </tr>
    <tr>
      <th>429</th>
      <td>9.060331</td>
      <td>7.467371</td>
      <td>8.183118</td>
      <td>3.850148</td>
      <td>4.430817</td>
      <td>7.824446</td>
      <td>429</td>
    </tr>
    <tr>
      <th>439</th>
      <td>7.932721</td>
      <td>7.437206</td>
      <td>7.828038</td>
      <td>4.174387</td>
      <td>6.167516</td>
      <td>3.951244</td>
      <td>439</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Detergents_Paper':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
      <td>75</td>
    </tr>
    <tr>
      <th>161</th>
      <td>9.428190</td>
      <td>6.291569</td>
      <td>5.645447</td>
      <td>6.995766</td>
      <td>1.098612</td>
      <td>7.711101</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Delicatessen':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
      <td>66</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.248504</td>
      <td>9.724899</td>
      <td>10.274568</td>
      <td>6.511745</td>
      <td>6.728629</td>
      <td>1.098612</td>
      <td>109</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
      <td>128</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8.034955</td>
      <td>8.997147</td>
      <td>9.021840</td>
      <td>6.493754</td>
      <td>6.580639</td>
      <td>3.583519</td>
      <td>137</td>
    </tr>
    <tr>
      <th>142</th>
      <td>10.519646</td>
      <td>8.875147</td>
      <td>9.018332</td>
      <td>8.004700</td>
      <td>2.995732</td>
      <td>1.098612</td>
      <td>142</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
      <td>154</td>
    </tr>
    <tr>
      <th>183</th>
      <td>10.514529</td>
      <td>10.690808</td>
      <td>9.911952</td>
      <td>10.505999</td>
      <td>5.476464</td>
      <td>10.777768</td>
      <td>183</td>
    </tr>
    <tr>
      <th>184</th>
      <td>5.789960</td>
      <td>6.822197</td>
      <td>8.457443</td>
      <td>4.304065</td>
      <td>5.811141</td>
      <td>2.397895</td>
      <td>184</td>
    </tr>
    <tr>
      <th>187</th>
      <td>7.798933</td>
      <td>8.987447</td>
      <td>9.192075</td>
      <td>8.743372</td>
      <td>8.148735</td>
      <td>1.098612</td>
      <td>187</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.368187</td>
      <td>6.529419</td>
      <td>7.703459</td>
      <td>6.150603</td>
      <td>6.860664</td>
      <td>2.890372</td>
      <td>203</td>
    </tr>
    <tr>
      <th>233</th>
      <td>6.871091</td>
      <td>8.513988</td>
      <td>8.106515</td>
      <td>6.842683</td>
      <td>6.013715</td>
      <td>1.945910</td>
      <td>233</td>
    </tr>
    <tr>
      <th>285</th>
      <td>10.602965</td>
      <td>6.461468</td>
      <td>8.188689</td>
      <td>6.948897</td>
      <td>6.077642</td>
      <td>2.890372</td>
      <td>285</td>
    </tr>
    <tr>
      <th>289</th>
      <td>10.663966</td>
      <td>5.655992</td>
      <td>6.154858</td>
      <td>7.235619</td>
      <td>3.465736</td>
      <td>3.091042</td>
      <td>289</td>
    </tr>
    <tr>
      <th>343</th>
      <td>7.431892</td>
      <td>8.848509</td>
      <td>10.177932</td>
      <td>7.283448</td>
      <td>9.646593</td>
      <td>3.610918</td>
      <td>343</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'sn':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


*** Outliars data***

Outliars indices 


```python
[i for i, n in bad_indexes.items() if n > 1]
```




    [128, 154, 65, 66, 75]




```python
outliars_data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
      <th>sn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
      <td>128</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
      <td>154</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
      <td>65</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
      <td>66</td>
    </tr>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



#### Visualization of outliers 


```python
def outlier_plotter(Feature):
    ax = good_data.plot(kind ='scatter',\
               x = 'sn',\
               y = Feature,\
               color='LightGreen',\
               label='good_data',\
               s=30)
    outliars_data.plot(kind ='scatter', \
                   x = 'sn' ,\
                   y = Feature,\
                   color='Darkred',\
                   label='outliars_data',\
                   s=60,ax=ax)
```


```python
outlier_plotter(Feature='Milk')
outlier_plotter(Feature='Grocery')
outlier_plotter(Feature='Fresh')
outlier_plotter(Feature='Frozen')
good_data = good_data.drop('sn',axis =1)
```


![png](output_69_0.png)



![png](output_69_1.png)



![png](output_69_2.png)



![png](output_69_3.png)


---------

### Question 4
*Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

**Answer:**

Yes, There are data points considered outliers for more than one feature based on the definition above.

Yes, they should be removed  and the reason is that the data points at row [128, 154, 65, 66, 75] has more than one outlier features because they are more likely to be true outliers than others.

Removing all of them could cause underfitting. So, for ones with one feature outlier, I decided to keep them as single outlier feature. It may be the desirable pattern in the datasets.



--------

## Feature Transformation
In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

### Implementation: PCA

Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.

In the code block below, you will need to implement the following:
 - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
from sklearn.decomposition import PCA


# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)
```


![png](output_76_0.png)



```python
display(pca_results)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Explained Variance</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dimension 1</th>
      <td>0.4430</td>
      <td>0.1675</td>
      <td>-0.4014</td>
      <td>-0.4381</td>
      <td>0.1782</td>
      <td>-0.7514</td>
      <td>-0.1499</td>
    </tr>
    <tr>
      <th>Dimension 2</th>
      <td>0.2638</td>
      <td>-0.6859</td>
      <td>-0.1672</td>
      <td>-0.0707</td>
      <td>-0.5005</td>
      <td>-0.0424</td>
      <td>-0.4941</td>
    </tr>
    <tr>
      <th>Dimension 3</th>
      <td>0.1231</td>
      <td>-0.6774</td>
      <td>0.0402</td>
      <td>-0.0195</td>
      <td>0.3150</td>
      <td>-0.2117</td>
      <td>0.6286</td>
    </tr>
    <tr>
      <th>Dimension 4</th>
      <td>0.1012</td>
      <td>-0.2043</td>
      <td>0.0128</td>
      <td>0.0557</td>
      <td>0.7854</td>
      <td>0.2096</td>
      <td>-0.5423</td>
    </tr>
    <tr>
      <th>Dimension 5</th>
      <td>0.0485</td>
      <td>-0.0026</td>
      <td>0.7192</td>
      <td>0.3554</td>
      <td>-0.0331</td>
      <td>-0.5582</td>
      <td>-0.2092</td>
    </tr>
    <tr>
      <th>Dimension 6</th>
      <td>0.0204</td>
      <td>0.0292</td>
      <td>-0.5402</td>
      <td>0.8205</td>
      <td>0.0205</td>
      <td>-0.1824</td>
      <td>0.0197</td>
    </tr>
  </tbody>
</table>
</div>



```python
print pca_results['Explained Variance'].cumsum()
```

    Dimension 1    0.4430
    Dimension 2    0.7068
    Dimension 3    0.8299
    Dimension 4    0.9311
    Dimension 5    0.9796
    Dimension 6    1.0000
    Name: Explained Variance, dtype: float64


-------

### Question 5
*How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
**Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the indivdual feature weights.


**Answer:**

1. **How much variance in the data is explained in total by the first and second principal component?**

    Total variance by the first and second principal component: ***0.7068***

2. **What about the first four principal components?**

    Total variance by first four princile components: ***0.9311***

3. **Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.**

    ***Dimension 1:***  It shows large increases for features Milk, Grocery and Detergents_Paper, a small increase for Delicatessen, and small decreases for features Fresh and Frozen.
    
    ***Dimension 2:***  It shows large increases for Fresh, Frozen and Delicatessen, and small increase for Milk, Grocery and Detergents_Paper.
    
    ***Dimension 3:***  It shows large increases for Frozen and Delicatessen, and large decreases for Fresh and Detergents_Paper.
    
    ***Dimension 4:***  It shows large increases for Frozen and Detergents_Paper, and large a decrease for Fish and Delicatessen.

--------

### Observation
Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.


```python
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.2406</td>
      <td>-1.2419</td>
      <td>-1.0729</td>
      <td>-1.9589</td>
      <td>0.2160</td>
      <td>-0.1782</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7394</td>
      <td>-2.9834</td>
      <td>-0.8204</td>
      <td>1.2945</td>
      <td>0.1297</td>
      <td>0.4712</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.1528</td>
      <td>5.3859</td>
      <td>0.0930</td>
      <td>0.4023</td>
      <td>0.3577</td>
      <td>0.3111</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Dimensionality Reduction
When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.

In the code block below, you will need to implement the following:
 - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
```

### Observation
Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.


```python
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# scatter plot of reduced features
#pd.scatter_matrix(reduced_data, alpha = 0.3, figsize = (10,10), diagonal = 'kde');
sns.pairplot(reduced_data,size =5)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.2406</td>
      <td>-1.2419</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7394</td>
      <td>-2.9834</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.1528</td>
      <td>5.3859</td>
    </tr>
  </tbody>
</table>
</div>





    <seaborn.axisgrid.PairGrid at 0x119715710>




![png](output_88_2.png)


## Visualizing a Biplot
A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.

Run the code cell below to produce a biplot of the reduced-dimension data.


```python
# Create a biplot
vs.biplot(good_data, reduced_data, pca)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119c34b10>




![png](output_90_1.png)


### Observation

Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 

From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

----------------

## Clustering

In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

--------

### Question 6
*What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

**Answer:**

1. **What are the advantages to using a K-Means clustering algorithm?**

     It is simple to understand and easy to implement.In computational point of view, it is  fast to run and always converge. It is suitable for searching convex clusters.
     
2. **What are the advantages to using a Gaussian Mixture Model clustering algorithm?**

  It is the fastest algorithm for learning mixture models among other available models. In case of overlapping clusters, it has "soft" classification technique available. In GMM, there are well-studied statistical inference techniques available.

3. **Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?**
    
    ***Gaussian Mixture Model clustering algorithm*** would be the most appropriate because from preious scatter graph of data sets with reduced features, there are two clusters that overlap with each other. 

------

### Implementation: Creating Clusters
Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.

In the code block below, you will need to implement the following:
 - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
 - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
 - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
 - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
 - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
   - Assign the silhouette score to `score` and print the result.


```python
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score

# method to find the best n_components
def score_cluster(data, num_components):
    clusterer = GMM(n_components=num_components)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    score = silhouette_score(data, preds)
    return score
    
print ("Silhouette Score for different sizes")
silhouette_scores_matrix = pd.DataFrame(index=['Score'])

for size in range(2,11):
    silhouette_scores_matrix[size] = pd.Series(score_cluster(reduced_data, size),\
                                               index = silhouette_scores_matrix.index)
    
display(silhouette_scores_matrix)

best_n_components = 2

# Apply the selected clustering algorithm to the reduced data 
clusterer = GMM(n_components=best_n_components)
clusterer.fit(reduced_data)

# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# Find the cluster centers
centers = clusterer.means_

# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)

print("Best Score: {}, n_components={}".format(score, best_n_components))
```

    Silhouette Score for different sizes



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Score</th>
      <td>0.421917</td>
      <td>0.395557</td>
      <td>0.279418</td>
      <td>0.321519</td>
      <td>0.25418</td>
      <td>0.323427</td>
      <td>0.309646</td>
      <td>0.328185</td>
      <td>0.319549</td>
    </tr>
  </tbody>
</table>
</div>


    Best Score: 0.422324682646, n_components=2



```python
silhouette_scores_matrix.plot(kind = "barh",figsize=(14,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11ca59050>




![png](output_100_1.png)


-------

### Question 7
*Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

**Answer:**

It is shown in previous output of "**Implementation: Creating Clusters**". the Best Score: 0.42, for number of components=2.

---------

### Cluster Visualization
Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 


```python
# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)
```


![png](output_106_0.png)


### Implementation: Data Recovery
Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.

In the code block below, you will need to implement the following:
 - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
 - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.



```python
# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments

display(true_centers)
#show segments in percentile
newdata = data.append(true_centers)


print ("Percentiles of the centers")
percent_centers = 100.0 * newdata.rank(axis=0, pct=True)\
                         .loc[['Segment 0', 'Segment 1']]\
                         .round(decimals=3)
        
display(percent_centers)

```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>8939.0</td>
      <td>2108.0</td>
      <td>2758.0</td>
      <td>2073.0</td>
      <td>352.0</td>
      <td>730.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>3567.0</td>
      <td>7860.0</td>
      <td>12249.0</td>
      <td>873.0</td>
      <td>4713.0</td>
      <td>966.0</td>
    </tr>
  </tbody>
</table>
</div>


    Percentiles of the centers



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>52.5</td>
      <td>34.4</td>
      <td>34.4</td>
      <td>58.4</td>
      <td>32.0</td>
      <td>41.2</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>28.1</td>
      <td>79.0</td>
      <td>80.5</td>
      <td>31.0</td>
      <td>80.3</td>
      <td>50.2</td>
    </tr>
  </tbody>
</table>
</div>


#### Plot of true center


```python
true_centers.plot(kind = "barh",figsize=(10,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1196ac210>




![png](output_110_1.png)


#### Plot of Percentile centers


```python
percent_centers.plot(kind = "barh",figsize=(10,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b6d8950>




![png](output_112_1.png)


--------

### Question 8
Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
**Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

**Answer:**

- **Segment 0** customers buy most Fresh, Frozen (> %50) and much Delicatessen (40%), it's mostly likely to be a **convinence store or greegrocery**.

- **Segment 1** customers buy lots of Grocery, Milk and Detergents_Paper (> 70%), following by Delicatessen (~%50), it indicates it's a **supermarket**.

---------

### Question 9
*For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*

Run the code block below to find which cluster each sample point is predicted to be.


```python
# Display the predictions
for i, pred in enumerate(sample_preds):
    print ("Sample point", i, "predicted to be in Cluster", pred)
```

    Sample point 0 predicted to be in Cluster 0
    Sample point 1 predicted to be in Cluster 1
    Sample point 2 predicted to be in Cluster 0



```python
samples.plot.barh(stacked=True,figsize=(10,2) );
```


![png](output_119_0.png)


**Answer:**

1. **For each sample point, which customer segment from Question 8 best represents it?**

    It is shown in output of **Question 9**.

2. **Are the predictions for each sample point consistent with this?**

    Yes, more or less. The prediction agrees with previous guess that samples are Supermarket, Greengrocer, except for sample 3 as Convenience store.

    Sample 1 customers buy lots of produces in all category, which is close to Segment 0.
    
    Sample 2 customers buy mostly Fresh and frozen, which makes it close to Segment 1.
    
    Sample 3 customers buy mostly Grocery and Detergents_Paper, which makes it close to Segment 0.


- Samples

| Fresh| Milk | Grocery | Frozen | Detergents_Paper | Delicatessen
--- | --- | --- | --- | --- | ---
0|	1.560499|	0.884800|	0.400925|	-0.574313|	0.209873|	0.499176
1|	5.084854|	-0.315148|	-0.089470|	2.776994|	-0.441685|	-0.215439
2|	-0.942242|	-0.297242|	0.468664|	-0.613289|	0.018584|	-0.519319


------------

## Conclusion

In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

----------

### Question 10
Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively.

**How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?**

**Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

**Answer:**

The items that customer purchase like: Fresh, Milk and Frozen are time sensitive and customer may prefer faster and frequent dilivery. Clustering designed with focusing on these three features may give  distributor more intution.

 One can perform the A/B test by selecting a random sample from each cluster and reducing the delivery frequency for each of them. Then a suvey of customer satisfaction could be made for further analysis. An equally sized group of customers with same shopping conditions except the delivery time selected from the remaining customers could be used for comparison. The A/B test data is prepared, interpreated and cross-validated by changing the delivery service to certain amount of test customers. The process can be iterated until certain model is verified or market goals are met.

---------

### Question 11
Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service. 

**How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?***  
**Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

**Answer:**

The customer segment (cluster prediction) can be used as target and one can design the supervised learning model based on them. The model developed could be used to predict the new customer data. 

-------

### Visualizing Underlying Distributions

At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.

Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.


```python
# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)
```


![png](output_132_0.png)


-------

### Question 12
*How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

**Answer:**

1. **How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers?** 
    
   The algorithm and the number of clusters chosen matches with the underlying distribution.
   
2. **Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution?**

    No, there will be overlap areas for two clusters. Although there is some overlap between the two groups in the middle of the distribution, segment 0 in the analysis is clearly "Retail", and segment 1 is clearly "Hotel/Restaurants/Cafes".

3. **Would you consider these classifications as consistent with your previous definition of the customer segments?**

    It shows the previous definition of customer segments is consistent with the classification using the Channel feature. I believe the algorithm did a reasonable job of clustering these customers according to broad categories of business type.

-------

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.


```python

```
