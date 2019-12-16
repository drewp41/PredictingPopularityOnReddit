<img src="https://miro.medium.com/max/3840/1*e3E0OQzfYCuWk0pket5dAA.png" alt="Reddit Logo" />
<center> <h1 style="font-size:36px;">Predicting Upvotes and Popularity on Reddit</h1> </center>
<h4>
Authors: Andrew Paul, Chigozie Nna</h4>
<hr>
<body>
<h1>Introduction </h1>
    
Reddit is an American social news aggregation, web content rating, and discussion website. Reddit originally created by two University of Virgina Students, Steven Huffman and Alexis Ohanian, in the year 2005. A year later Condé Nast Publications acquired the site as their own. Popularity in Reddit began to arise, as by 2007 NSFW, Programming, and Science where the the top trending subreddits of the time. By the year 2008, a launch of numerous different subreddits began to popularize the site, with Reddit being able to gain enough popularity to overtake Digg in search popularity by 2010. Reddit’s rise to fame did not stop there with, Reddit finally achieving a total of one billion page views per month in the year 2011. The goal of Reddit is for members to be able to submit content to the site in the form of links, text posts, and images, which can then be voted up or down by opposing members. The posts are categorized into items called “Subreddits” where users can share specific topics and/or interests that relate to the category at hand. Full details on it’s timeline and history can be viewed here.

In this tutorial, our goal is to tidy up the data of posts within a years total, to provide us with knowledge into which what the amount of characters in a post cause the most effect in terms of up votes, down votes, score, and in general a reaction to the post. Post may vary in topics, arguments, time posted, and many more varieties, but we feel as if the popularity really depends on the length of characters used. We will be able to determine which length is just to short, and what length is long enough to bore an audience and not give the time to react to it. We hope to give enough information and analysis to provide, clarity, understanding and hopefully a new found interest to readers that are unfamiliar with the social foreground. Hopefully those who are frequent Reddit users, will gain some insight on how long they should make their posts if they are trying to gain more popularity.
<body>
<hr>
<body>
<h1 id="getting-started-with-the-data">Getting started with the Data</h1>
<p>We decided to use Python 3 and SQL to help gain and analyze our data. Crucial libraries used to help us where: <a href="https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673">pandas</a>, <a href="https://matplotlib.org/">matplotlib</a>, <a href="https://python-graph-gallery.com/seaborn/">seaborn</a>, and <a href="https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/">scikit-learn</a>.</p>
</body>


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

<body> We plan on using multiple panda dataframes that will be read in using SQL commands through googles BigQuery website.
    
<h2>Processing and Recieving data </h2>

We used the following SQL command through Googles BigQuery to at first take data from a third party called <a href=https://pushshift.io/>"Pushshift"</a> that is a Reddit API that tracks almost all of Reddit's for the last few years. We are taking in data from 2016 to Augst 2019 due to the immense amount of data that is tracked.
</body>

<img src= "https://i.imgur.com/xc6mlpA.png>" alt= "SQL Code" width="400"/>

<body> In this SQL Query we are getting the length of every single title, averaging the score based on the length of the title, the average number of comments based on the length of the title, and the number of posts with that amount of characters. This is done by using the 'GROUP BY' command with SQL. BigQuery convertd this data into a <a href= https://www.howtogeek.com/348960/what-is-a-csv-file-and-how-do-i-open-it/>csv file</a>, which is a table or excel seperating the data by commas (,) making it easy to parse and split the data with.

<h2> Reading the Data </h2>

We will First use Pythons Pandas to read in the csv file and convert it into a panda <a href=https://www.geeksforgeeks.org/python-pandas-dataframe/>dataframe</a>, which is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)


```python
data = pd.read_csv('LengthScoreComments.csv', sep=',')
data[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length_title</th>
      <th>avg_score</th>
      <th>avg_comments</th>
      <th>num_posts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>40.804163</td>
      <td>1.789499</td>
      <td>472705</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>65.521526</td>
      <td>2.796440</td>
      <td>739424</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>65.577614</td>
      <td>3.039975</td>
      <td>1361269</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>92.408734</td>
      <td>3.559541</td>
      <td>2588850</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>84.223042</td>
      <td>3.203536</td>
      <td>2210213</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>129.382588</td>
      <td>3.370371</td>
      <td>3850745</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>105.844544</td>
      <td>3.985902</td>
      <td>2858907</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>99.618349</td>
      <td>4.208025</td>
      <td>2939891</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>104.936504</td>
      <td>4.531499</td>
      <td>3818682</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>103.995719</td>
      <td>4.758954</td>
      <td>3773035</td>
    </tr>
  </tbody>
</table>
</div>



In the DataFrame above you can see: 
   * length_title: Amount of Characters in the title
   * avg_score: The average Score the post will recieve with character length
   * avg_comments: The average amount of comments a post will recieve with character length
   * num_posts: The Number of posts between 2016-Aug 2019 with character Length

<hr size="20">
<body>
<h2> Graphing</h2>

In this first graph we will graph to see the relation between Length of Title verse the Average Score to see if there is a relation between if a reddit user will reciever more votes based on the character length of their post. This can help readers get an insight to how long they should make their posts if they desire to be the most popular reddit user of their peers.
</body>


```python
plt.scatter(x = data['length_title'], y = data['avg_score'], s = data['num_posts']/200000)
plt.title('Length of Post Title vs Average Score of Post')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()
```


![png](Final_Project_Code_files/Final_Project_Code_7_0.png)


<body> We can see that there is a clear relation between the upvotes and the  number of characters in the post title. Based on the graph it seems that it is the best to have captions of Character lengths between 5-25 and 153-300+. The range most likely is based off the fact that the type of posts are farely different. There are many popular subreddits named short and quick things like "meow" to follow a trend that will be a picture of a cat that revieve many likes as long as they are following the trend. This Explains the peak and the downtrend of the number of likes as the posts begin to be normal and causal day to day type of posts. The number of UpVotes however does begin to rise again as the number of character are longer. This is because as the characters get longer, they tend to be actual issues and problems that recieve more views and reactions (ex: A president trump quote = more characters and responses).
<body>

<hr size="20">

<h2> Comments </h2>

<p> We then decided to check if there was a relation between amount of comments verse the character length of the posts. People will place upvotes to anything they think is funny, however we wanted to see which posts actually get people commenting. Comments are what we deemed as true reactions to the posts, since it requires viewers to put in more effort than just a click for an upvote.</p>


```python
plot = plt.scatter(x = data['length_title'], y = data['avg_comments'], s = data['num_posts']/200000)
plt.title('Avg Number of Comments vs Length of Post title')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Comments')
plt.show()
```


![png](Final_Project_Code_files/Final_Project_Code_9_0.png)


<body>Based on the relation we are able to see that the Number of comments are actually greater based on the character length of the posts. A reasoning for this data is that the longer the character length in posts most likely mean, it is on a controversial topic, quote, or a scientific analysis that people are more likely to comment on and put their input. This is in contrast to small memes and jokes that will mostly recieve likes and not comments.</body>

<hr size=20>

<center> <h2> Reddit Artwork </h2> </center>
<img src="https://static.makeuseof.com/wp-content/uploads/2019/04/whats-reddit-670x335.jpg" />


<h2> Filtered Data </h2>
    
<p> We then decided to filter our data because there was very big bias with the top subreddits with small captions. We want to see the relationship between length and upvotes for typical reddit day to day users, that are making posts that arent silly small trends like "meow"</p>

<img src="https://i.imgur.com/bALsUPt.png" width="500" />

<p> In this SQL Query we are creating an entirely new cvs file with the top 15 subreddits to see the relation between these subreddits that people typically talk about on a day to day basis. This data should give an accurate analysis of what we were looking for and give a different result in the relation shown in the graph<p>


```python
Unbias = pd.read_csv('FilteredSubLengthScore.csv', sep=',')
Unbias[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length_title</th>
      <th>avg_score</th>
      <th>num_posts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>90.522604</td>
      <td>9401</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>210.944914</td>
      <td>18934</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>207.167737</td>
      <td>35812</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>218.983280</td>
      <td>61842</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>241.415093</td>
      <td>57298</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>271.754140</td>
      <td>62556</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>285.368175</td>
      <td>77455</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>288.794141</td>
      <td>94686</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>316.807680</td>
      <td>123414</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>313.843326</td>
      <td>141357</td>
    </tr>
  </tbody>
</table>
</div>



In the DataFrame above you can see: 
   * length_title: Amount of Characters in the title
   * avg_score: The average Score the post will recieve with character length
   * num_posts: The Number of posts between 2016-Aug 2019 with character Length


<hr size="20">
<body>
<h2> Graphing Filtered Data</h2>

In this next graph we will graph to see the relation between Length of Title verse the Average Score to see if there is the same relation as the first graph, however if there is now a change in the data now that it is based off the top 15 most popular subreddits. This should give a more accurate amount of data that should actually help reddit users decide how loong their pots should be so they can be popular.
</body>


```python
plt.scatter(x = Unbias['length_title'], y = Unbias['avg_score'], s = Unbias['num_posts']/10000)
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()
```


![png](Final_Project_Code_files/Final_Project_Code_15_0.png)


<p>Based on the new graph we can see a less skewed range of values and a smaller drop off between the relationship. We can see a more accurate drop off. It seems that the amount of characters required are still the same despite the data being filtered now. This overall proves the fact that it is best to have posts that are between the amounts of 5-25 characters. </p>

<hr size="20">

<center> <h2> Linear & Polynomial Regression </h2> </center>

<p> We decied that we would like to get a relative prediction of what the outcome would be by create first a linear regression of the data and then create a ploynomial trend line to the data since it is not a linear change based on the look. </p>

<h3> Linear </h3>


```python
Y = Unbias['avg_score']
X = Unbias['length_title']

linear_regression = LinearRegression()
linear_regression.fit(X.values.reshape(-1, 1), Y)
model = linear_regression.predict(X.values.reshape(-1,1))

plt.figure(figsize=(10,8));
plt.scatter(X, Y);
plt.plot(X, model);
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
```




    Text(0, 0.5, 'Average Score of Post')




![png](Final_Project_Code_files/Final_Project_Code_17_1.png)


<h3> Polynomial </h3>


```python
poly_reg = PolynomialFeatures(degree=2)
poly = poly_reg.fit_transform(X.values.reshape(-1, 1))


linear_regression2 = LinearRegression()
linear_regression2.fit(poly, Y.values.reshape(-1, 1))
y_pred = linear_regression2.predict(poly)
plt.figure(figsize=(10,8));
plt.scatter(X, Y);
plt.plot(X, y_pred);
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
```




    Text(0, 0.5, 'Average Score of Post')




![png](Final_Project_Code_files/Final_Project_Code_19_1.png)


<p>As you can see the ploynomial regression gives us a more accurate analysis of the data. We can see, the relation between the length of characters in a post and the average number of upvotes is a clear polynomial relation. Reddit users can use this now to get an accurate way to decide if they want to either go with a short amount of characters to get around 300 upvotes, or go with a more lengthy response to get 300+ upvotes. </p>
<hr size=20>

<center> <h2> Typical Reddit Home Page </h2></center>
<img src="https://i.redd.it/vb63xmw7skm21.png" width= 800/>


<center> <h2> Time Matters </h2> </center> 

<p>In this section we will be analyzing whether the time and day of the post also come in as a factor to upvotes. This way us Reddit users can pinpoint the very day and time we should post to get the maximum popularity and be seen as a legend to our peers. We first started by getting new data again through BigQuery</p>

<img src="https://i.imgur.com/63bqeZ2.png" width= 700/>

<p> Relationship between time of day posted and average score of the post (if score >= 100). We looked at the average score of posts on every hour of every day of the week, so 24 hours in a day x 7 days a week gives us 168 rows of data points </p>



```python
timeScore = pd.read_csv('TimeVsScore.csv', sep=',')
formattedTimeScore = timeScore.copy()
formattedTimeScore['hourofday'] = formattedTimeScore['hourofday'].apply(convertHourToTime)
formattedTimeScore['dayofweek'] = formattedTimeScore['dayofweek'].apply(convertNumToDay)

# Converts the numbers 0-23 to their respective times
def convertHourToTime(num):
    if num == 0:
        timeOfDay = '12 AM'
    elif num <= 11:
        timeOfDay = str(num) + ' AM'
    elif num == 12:
        timeOfDay = '12 PM'
    else: 
        timeOfDay = str(num-12) + ' PM'
    return timeOfDay

# Converts the numbers 1-7 to their respective weekdays
def convertNumToDay(num):
    weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    return weekdays[num-1]

formattedTimeScore
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_score</th>
      <th>dayofweek</th>
      <th>hourofday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>817.979886</td>
      <td>Sunday</td>
      <td>12 AM</td>
    </tr>
    <tr>
      <td>1</td>
      <td>832.602126</td>
      <td>Sunday</td>
      <td>1 AM</td>
    </tr>
    <tr>
      <td>2</td>
      <td>926.316992</td>
      <td>Sunday</td>
      <td>2 AM</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1013.444329</td>
      <td>Sunday</td>
      <td>3 AM</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1134.617892</td>
      <td>Sunday</td>
      <td>4 AM</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>163</td>
      <td>971.682509</td>
      <td>Saturday</td>
      <td>7 PM</td>
    </tr>
    <tr>
      <td>164</td>
      <td>942.749931</td>
      <td>Saturday</td>
      <td>8 PM</td>
    </tr>
    <tr>
      <td>165</td>
      <td>911.208568</td>
      <td>Saturday</td>
      <td>9 PM</td>
    </tr>
    <tr>
      <td>166</td>
      <td>880.878130</td>
      <td>Saturday</td>
      <td>10 PM</td>
    </tr>
    <tr>
      <td>167</td>
      <td>823.783345</td>
      <td>Saturday</td>
      <td>11 PM</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 3 columns</p>
</div>




```python
timeScoreMatrix = timeScore.pivot(index='dayofweek', columns='hourofday', values='avg_score')
cols = timeScoreMatrix.columns.tolist()
cols.insert(0, cols.pop(cols.index(23)))
fixedTimeScoreMatrix = timeScoreMatrix.reindex(columns=cols)
timeScoreMatrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>hourofday</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>dayofweek</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>817.979886</td>
      <td>832.602126</td>
      <td>926.316992</td>
      <td>1013.444329</td>
      <td>1134.617892</td>
      <td>1283.984606</td>
      <td>1400.718513</td>
      <td>1384.293555</td>
      <td>1339.667450</td>
      <td>1227.160091</td>
      <td>...</td>
      <td>940.857548</td>
      <td>968.624444</td>
      <td>937.781498</td>
      <td>929.255882</td>
      <td>967.731765</td>
      <td>942.091881</td>
      <td>964.019933</td>
      <td>963.388646</td>
      <td>853.194138</td>
      <td>802.564590</td>
    </tr>
    <tr>
      <td>2</td>
      <td>815.127267</td>
      <td>862.858846</td>
      <td>928.686203</td>
      <td>1042.522746</td>
      <td>1189.814349</td>
      <td>1386.860338</td>
      <td>1369.992439</td>
      <td>1293.379859</td>
      <td>1220.160733</td>
      <td>1125.224842</td>
      <td>...</td>
      <td>957.458976</td>
      <td>938.846684</td>
      <td>938.656955</td>
      <td>939.169165</td>
      <td>937.429680</td>
      <td>957.803950</td>
      <td>937.148762</td>
      <td>886.015139</td>
      <td>810.725206</td>
      <td>800.888918</td>
    </tr>
    <tr>
      <td>3</td>
      <td>807.148566</td>
      <td>859.623517</td>
      <td>974.795660</td>
      <td>1067.599219</td>
      <td>1179.800492</td>
      <td>1357.187672</td>
      <td>1392.787999</td>
      <td>1288.674100</td>
      <td>1203.942667</td>
      <td>1137.456890</td>
      <td>...</td>
      <td>968.803341</td>
      <td>962.026646</td>
      <td>934.731859</td>
      <td>944.079777</td>
      <td>949.844017</td>
      <td>973.267883</td>
      <td>955.275513</td>
      <td>890.547645</td>
      <td>838.271816</td>
      <td>822.564901</td>
    </tr>
    <tr>
      <td>4</td>
      <td>820.053926</td>
      <td>864.788059</td>
      <td>966.607819</td>
      <td>1086.004091</td>
      <td>1201.194851</td>
      <td>1373.418265</td>
      <td>1386.699401</td>
      <td>1285.982900</td>
      <td>1201.614869</td>
      <td>1116.236775</td>
      <td>...</td>
      <td>969.912780</td>
      <td>957.093768</td>
      <td>941.838205</td>
      <td>947.765502</td>
      <td>929.672191</td>
      <td>966.943421</td>
      <td>925.359047</td>
      <td>874.487371</td>
      <td>815.822274</td>
      <td>818.863226</td>
    </tr>
    <tr>
      <td>5</td>
      <td>810.921800</td>
      <td>883.710208</td>
      <td>969.234269</td>
      <td>1078.066438</td>
      <td>1207.298409</td>
      <td>1366.675789</td>
      <td>1366.740784</td>
      <td>1282.025036</td>
      <td>1212.700262</td>
      <td>1112.079900</td>
      <td>...</td>
      <td>958.369804</td>
      <td>932.961184</td>
      <td>935.878545</td>
      <td>949.448003</td>
      <td>930.227652</td>
      <td>964.164243</td>
      <td>938.896549</td>
      <td>885.560502</td>
      <td>840.863545</td>
      <td>813.365210</td>
    </tr>
    <tr>
      <td>6</td>
      <td>824.531597</td>
      <td>883.380917</td>
      <td>946.547062</td>
      <td>1009.453494</td>
      <td>1188.025956</td>
      <td>1347.901959</td>
      <td>1383.533918</td>
      <td>1283.798372</td>
      <td>1211.544344</td>
      <td>1119.912744</td>
      <td>...</td>
      <td>943.259381</td>
      <td>940.068199</td>
      <td>915.703810</td>
      <td>920.797151</td>
      <td>924.610139</td>
      <td>937.115934</td>
      <td>920.370725</td>
      <td>894.035268</td>
      <td>857.822298</td>
      <td>790.034420</td>
    </tr>
    <tr>
      <td>7</td>
      <td>799.677349</td>
      <td>843.552096</td>
      <td>933.552917</td>
      <td>1037.488966</td>
      <td>1152.136520</td>
      <td>1315.464118</td>
      <td>1392.310930</td>
      <td>1390.827688</td>
      <td>1304.164204</td>
      <td>1162.523854</td>
      <td>...</td>
      <td>947.246423</td>
      <td>940.004676</td>
      <td>945.921120</td>
      <td>943.630010</td>
      <td>936.120690</td>
      <td>971.682509</td>
      <td>942.749931</td>
      <td>911.208568</td>
      <td>880.878130</td>
      <td>823.783345</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 24 columns</p>
</div>




```python
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl

fig = plt.figure()
fig, ax = plt.subplots(1,1,figsize=(15,15))
heatmap = ax.imshow(fixedTimeScoreMatrix, cmap='BuPu')
ax.set_xticklabels(np.append('', formattedTimeScore.hourofday.unique())) # columns
ax.set_yticklabels(np.append('', formattedTimeScore.dayofweek.unique())) # index

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_title("Time of Day Posted vs Average Score")
ax.set_xlabel('Time of Day Posted (EST)')
ax.set_ylabel('Day of Week Posted')

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", "3%", pad="1%")
fig.colorbar(heatmap, cax=cax)
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](Final_Project_Code_files/Final_Project_Code_24_1.png)


<p>As we can see from the data, the most popular times to post are between 6am-9am. 7am definitely has the highest ratio of upvotes over the seven days of the week, but it seems to be the most popular to make posts on Saturday and Sunday. After heavy analysis over all of these datasets we are finally able to make an accurate guess/prediction on what is best for Reddit users popularity in posts. Overall Reddit users should have the best ratios of upvotes if they post on a Saturday, at 7am, with roughly 5-25 characters in their post. This will give them the highest chance and possibility of getting a great amount of upvotes on their post, especially if it is a part of the top fifteen subreddits.

<hr size=20>

<center> <h2> Conclusion and More </h2> </center>

<p>Reddit has grown to be an outstanding social media website, however many still do not know the trick to nailing down how to get their posts viewed and seen by the public. This project helped us learn so much more about the website and what kind of data is continuaously being drawn by third party websites. As you can see from this analysis, there are many factors that can put into a post on reddit that will ultimately decide how many views and upvotes one will get. We hope that this helps new Reddit users get a great jump on how to begin posting and what kind of times and amount of characters they shoudl be using based on the topic.

If you are interested in Reddit and the many datasets to use, we recommend using Google's <a href =https://cloud.google.com/bigquery/docs/> BigQuery </a> and <a href=https://pushshift.io/>Pushshift</a>. This API consistently takes in data second by second, so we used only a select amount of data from 2016 to August of this year (2019). This tutorial was a minor fraction of the amount of data and things that can be done using Reddits data. We hope that others are inspired to do the same kind of tutorials, and we hope it was worth the read! </p>


```python

```
