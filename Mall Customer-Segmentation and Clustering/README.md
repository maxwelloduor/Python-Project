
## Introduction :
Customer Segmentation is the process of dividing a customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.
I will make use of K-means clustering which is an essential algorithm for clustering unlabelled datasets.

My aim is to cluster the given dataset into distinct target market groups and analyze the obtained target groups.This is in a bid to gain insight on target markets that may potentially lead to increased product sales at the mall when advertised to.

I will carry out my analysis using the 'Python Programming Language' on the Jupyter Notebook IDE.

## Dataset Overview
The dataset is acquired from kaggle and the link is given below :

https://www.kaggle.com/nelakurthisudheer/mall-customer-segmentation

The dataset consists of following five features of 200 customers:

- CustomerID: Unique ID assigned to the customer

- Gender: Gender of the customer

- Age: Age of the customer

- Annual Income (k$): Annual Income of the customer

- Spending Score (1-100): Score assigned by the mall based on customer behavior and spending nature

# Steps for Implementation

-import all necessary packages

```
import ----- from ------
import -----
```

## Univariate Analysis
1). For starters, I loaded the dataset and obtained the descriptive statistics of the dataset variables.

```
 df = pd.read_csv("C:/Users/user/Desktop/Downloads/Mall_Customers.csv")
 df.describe().transpose()
```
```
	CustomerID	Age	Annual Income (k$)	Spending Score (1-100)
count	200.000000	200.000000	200.000000	200.000000
mean	100.500000	38.850000	60.560000	50.200000
std	57.879185	13.969007	26.264721	25.823522
min	1.000000	18.000000	15.000000	1.000000
25%	50.750000	28.750000	41.500000	34.750000
50%	100.500000	36.000000	61.500000	50.000000
75%	150.250000	49.000000	78.000000	73.000000
max	200.000000	70.000000	137.000000	99.000000
```


2). Plots for each of the numerical variables.

Code:
```

columns =  ['Age', 'Annual Income (k$)','Spending Score (1-100)']

for i in columns:

    sns.kdeplot(data = df, x=i, shade = True, hue = df['Gender'])

```

Insights:


![download](https://github.com/maxwelloduor/Python-Project/assets/137492526/1aab6f44-2968-4b1d-bb75-b9339d92f0dd)


From the 'Age' kde plot, I observed that the age range of the customers is between 18-69 whereby most customers are in the age range 18-52. The female customers are more in number within the age range of about 18 to 58.In the 0-18 and 58-80 age ranges,the male gender customers are more in number.


![data3](https://github.com/maxwelloduor/Python-Project/assets/137492526/5a0d8942-36e2-40a7-85e4-2bfc7574f0c2)


From the 'Annual Income' kde plot,I observed that most of the customers lie between the 50k-80k annual income range while the total range lies between 18k to 52k.The female customers earn more annual income than the male customers within the income range 0-130k while in the 130-150k range the male customers outshine the female customers.


![data](https://github.com/maxwelloduor/Python-Project/assets/137492526/ec809f11-d51c-4d28-b1e2-6a9edd9b98e2)

From the visualization you can tell that female customers have a significantly higher spending score as compared to their male counterparts.



3). Boxplots for each numerical variable(categorized by gender).

Code:

```
columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']

for i in columns:

plt.figure()

sns.boxplot(data=df,x='Gender',y=df[i])

```

I was looking for outliers, and from the boxplots, I observed that there  is one outlier in the 'Annual income' visualizaton.


![data4](https://github.com/maxwelloduor/Python-Project/assets/137492526/b4958515-18db-4994-891a-0be1a41a5f77)



![data5](https://github.com/maxwelloduor/Python-Project/assets/137492526/b063a14d-0d65-429c-b007-a7e608ad1033)


![data6](https://github.com/maxwelloduor/Python-Project/assets/137492526/e8401067-a796-4b1b-8a14-44cea51f12a0)


5). Number of male and female customers in % form.

```

df['Gender'].value_counts(normalize=True)

```

```

Gender
Female    0.56
Male      0.44
Name: proportion, dtype: float64

```

The female customers outnumber the male customers by a 12% differnce.


## Bivariate Analysis.


1). Scatter plot to show the relationship between 'Annual Income and 'Spending score'.


Code: 

```

sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )

```

![data7](https://github.com/maxwelloduor/Python-Project/assets/137492526/2f3700f4-393e-42f0-a025-2c089a4f4800)

From the visualization, I observed that the two variables have a non-linear realtionship.


3) Correlation between the variables.

Code:

```
df.corr()
```

Output:

                  Age   	Annual Income (k$)	Spending Score (1-100)

                 Age	                     1.000000	-0.012398	-0.327227

                 Annual Income (k$)	    -0.012398	1.000000	0.009903

                 Spending Score (1-100)	-0.327227	0.009903	1.000000

From the output, I observed that 'Age' as a variable is negatively correlated to 'Annual income' and 'Spending score' variables.

While 'Annual income' is positively correlated with 'Spending score'.


4). Mean values of the numerical data variables,grouped by gender.


Code: 
```
df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()
```

Output:

                  Age	         Annual Income (k$)	   Spending Score (1-100)

     Gender

     Female	 38.098214	      59.250000	             51.526786

     Male	 39.806818	      62.227273	             48.511364

Although male customers have an averagely higher income though they spend less in comparision to female customers.



## Univariate Clustering.

### K-means Algorithm

- We specify the number of clusters that we need to create.

- The algorithm selects k objects at random from the dataset. This object is the initial cluster or mean.

- The closest centroid obtains the assignment of a new observation. We base this assignment on the 

Euclidean Distance between object and the centroid.

- k clusters in the data points update the centroid through calculation of the new mean values present in 

all the data points of the cluster. The kth cluster’s centroid has a - - Length of p that contains means 

of all variables for observations in the k-th cluster. We denote the number of variables with p.

- Iterative minimization of the total within the sum of squares. Then through the iterative minimization 

of the total sum of the square, the assignment stop wavering when we - - Achieve maximum iteration. The 

default value is 10 that the R software uses for the maximum iterations.


### Determining Optimal Clusters

While working with clusters, you need to specify the number of clusters to use. You would like to utilize 

the optimal number of clusters. To help you in determining the optimal clusters, there are three popular 

methods –

-- Elbow method The main goal behind cluster partitioning methods like k-means is to define the clusters 

such that the intra-cluster variation stays minimum.

minimize(sum W(Ck)), k=1…k


Where Ck represents the kth cluster and W(Ck) denotes the intra-cluster variation. With the measurement 

of the total intra-cluster variation, one can evaluate the compactness of the clustering boundary. We can 

then proceed to define the optimal clusters as follows –

First, we calculate the clustering algorithm for several values of k. This can be done by creating a 

variation within k from 1 to 10 clusters. We then calculate the total intra-cluster sum of square (iss).

Code:

```
intertia_scores=[]

for i in range(1,11):

kmeans=KMeans(n_clusters=i)

kmeans.fit(df[['Annual Income (k$)']])

intertia_scores.append(kmeans.inertia_)

```

Then, we proceed to plot iss based on the number of k clusters. This plot denotes the appropriate number 

of clusters required in our model. In the plot, the location of a bend or a knee is the indication of the 

optimum number of clusters.

Code:

```
 plt.plot(range(1,11),intertia_scores)
```

![data8](https://github.com/maxwelloduor/Python-Project/assets/137492526/72f94337-c81a-4936-aefa-6282285424d1)

I then initiated and fitted  the algorith and added an 'Income Cluster' column that contains clustered data values to the existing dataset.

```
 clustering1 = KMeans(n_clusters=3)

 clustering1.fit(df[['Annual Income (k$)']])

 df['Income Cluster'] = clustering1.labels

df.head()
```
```
	CustomerID	Gender	  Age	Annual Income (k$)	Spending Score (1-100)	Income Cluster
0	1	          Male	  19	     15	                   39	                 1
1	2	          Male    21         15	                   81                    1
2	3	          Female  20	     16                	   6	                 1
3	4	          Female  23	     16	                   77	                 1
4	5	          Female  31	     17                    40	                 1

```

### Bivariate Clustering.
i). Initiate the KMeans clustering algorithm

- fit the data values

- label the obtained clusters

- create new column

- obtain dataset overview.

Code:

```
clustering2 = KMeans()

clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])

df['Spending and Income Cluster'] =clustering2.labels_

df.head()

```
Output:

```
  CustomerID	Gender	  Age	  Annual Income (k$)	Spending Score (1-100)	Income Cluster	Spending and Income Cluster
0	1	Male	  19	   15	                  39	                   1	          4
1	2	Male	  21	   15	                  81	                   1	          1
2	3	Female	  20	   16	                  6	                   1	          4
3	4	Female	  23	   16	                  77	                   1	          1
4	5	Female	  31	   17	                  40	                   1	          4

```                       
To obtain the appropriate number of clusters for the dataset, I used the elbow method.

From the plot,I was able to obtain that the suitable number of clusters is 5.

Code:
```

intertia_scores2=[]

for i in range(1,11):

kmeans2=KMeans(n_clusters=i)

kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])

intertia_scores2.append(kmeans2.inertia_)

plt.plot(range(1,11),intertia_scores2)

```

![data9](https://github.com/maxwelloduor/Python-Project/assets/137492526/bf772642-451f-4f62-b68a-0eca1d51facf)


From the plot,I was able to obtain that the suitable number of clusters is 5.


A scatter plot to display the relationship between the 'Spending score' and 'Annual Income' with respect to Spending and Income Cluster

Code :
```
centers = pd.DataFrame(clustering2.cluster_centers_)

centers.columns = ['x', 'y']

plt.figure(figsize = (10,8))

plt.scatter(x = centers['x'], y = centers['y'], s = 100, c = 'black', marker = '*')

sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'Spending and

Income Cluster', palette = 'tab10')

```

![data10](https://github.com/maxwelloduor/Python-Project/assets/137492526/5bac1410-e92d-42d3-ace3-b1da95b6c158)



I Compared the  genders to their 'Spending and Income Cluster'.

Code:
```
pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize = 'index')
```
Output:
```
                                   Gender	Female	        Male
Spending and Income Cluster		
0	                                         0.538462	0.461538
1	                                         0.608696	0.391304
2	                                         0.457143	0.542857
3	                                         0.590909	0.409091
4	                                         0.592593	0.407407
```

I then grouped the variables by their 'Spending and Income' Clusters.

Code: 
```
df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()
```
Output:

```

	                             Age	  Annual Income (k$)	Spending Score (1-100)
Spending and Income Cluster			
0	                             32.692308	    86.538462	         82.128205
1	                             45.217391	    26.304348	         20.913043
2	                             41.114286	    88.200000	         17.114286
3	                             25.272727	    25.727273	         79.363636
4	                             42.716049	    55.296296	         49.518519
```

#### Reccommendations Obtained From The Project.
The aim of the project was to identify the best target customer group(s) in order for the marketing team to plan the best optimal strategy that ought to increase the mall's product sales.

Following my analysis, my reccommendations are as follows:

- The best target customer group is: Cluster 1.

This is because this cluster has both a high spending score and high income thus their able to purchase more products and at an often rate too.

54% of cluster 1 customers are women.The marketing team could run a marketing campaign advertising items that particularly resonate with this sub-cluster.

- Cluster 3 also presents an interesting opportunity to increase product sales.

Although this cluster is in the low income range,it is suprisingly also in the high spending score category.

The customers could potentially be inclined to purchase more products when popular items are discounted.
