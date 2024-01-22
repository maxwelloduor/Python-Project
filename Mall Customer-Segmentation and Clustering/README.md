# Mall Customer-Segmentation and Clustering
## Introduction :
Customer Segmentation is the process of dividing a customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.
I will make use of K-means clustering which is an essential algorithm for clustering unlabelled datasets.

My aim is to cluster the given dataset into distinct target market groups and analyze the obtained target groups.This is in a bid to gain insight on target markets that may potentially lead to increased product sales at the mall when advertised to.

I will carry out my analysis using the 'Python Programming Language' on the Jupyter Notebook IDE.

## Dataset Overview
The dataset is aquired from kaggle and the link is given below :

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



4). Boxplots for each numerical variable(categorized by gender).

Code:

```
columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']

for i in columns:

plt.figure()

sns.boxplot(data=df,x='Gender',y=df[i])

```

From the boxplots, I was looking for outliers. I observed that there  is one outliers in the 'Annual income' visualizaton.


![data4](https://github.com/maxwelloduor/Python-Project/assets/137492526/b4958515-18db-4994-891a-0be1a41a5f77)



![data5](https://github.com/maxwelloduor/Python-Project/assets/137492526/b063a14d-0d65-429c-b007-a7e608ad1033)


![data6](https://github.com/maxwelloduor/Python-Project/assets/137492526/e8401067-a796-4b1b-8a14-44cea51f12a0)


5). Number of male and female customers in % form.

```

df['Gender'].value_counts(normalize=True)

```

The female customers outnumber the male customers by a 12% differnce.


## Bivariate Analysis.


1). Scatter plot to show the relationship between 'Annual Income and 'Spending score'.

From the visualization, I observed that the two variables have a non-linear realtionship.

Code: sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )

image

2) Pairplots for the dataset variables.

From the pairplots, I observed the type of relationship each variable has with each other.

Code:

df=df.drop('CustomerID',axis=1)

sns.pairplot(df,hue='Gender')

image

3) Correlation between the variables.

From the output, I observed that the 'Age' variable is negatively correlated to the 'Annual income' and 'Spending score' variables.

While the 'Annual income' is positively correlated to the 'Spending score'.

Code: df.corr()

Output:

                  Age   	Annual Income (k$)	Spending Score (1-100)

                 Age	                     1.000000	-0.012398	-0.327227

                 Annual Income (k$)	    -0.012398	1.000000	0.009903

                 Spending Score (1-100)	-0.327227	0.009903	1.000000
4). Mean values of the numerical data variables,grouped by gender.

By observation, although the male customers have an averagely higher income though they spend less in comparision to the female customers.

Code: df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()

Output:

                Age	       Annual Income (k$)	   Spending Score (1-100)

     Gender

     Female	38.098214	      59.250000	           51.526786

     Male	 39.806818	      62.227273	             48.511364
5). Heatmap to display the correlation between variables.

To read the visualization, -1>x>0 indicates a negative correlatio between the variables while 0>x<1 indicates a positive correlation between the variables.

image

Univariate Clustering.
i) For loop to determine the intertia scores for clusters between the range of 1-11.

Code:

intertia_scores=[]

for i in range(1,11):

kmeans=KMeans(n_clusters=i)

kmeans.fit(df[['Annual Income (k$)']])

intertia_scores.append(kmeans.inertia_)
ii) Plot the intertia scores to obtain the most suitable cluster number,this is through the elbow method.

From the plot, I determined the appropriate lot size to be 3.

Code: plt.plot(range(1,11),intertia_scores)

image

iii) Initiate my algorithm.

Code: clustering1 = KMeans(n_clusters=3)
iv) Fit my algorithm into the data.

Code: clustering1.fit(df[['Annual Income (k$)']])
v) Add an 'Income Cluster' column that contains clustered data values to the existing dataset.

Code 1: df['Income Cluster'] = clustering1.labels and Code 2: df.head()

Output:

Gender	  Age	    Annual Income (k$)	Spending Score (1-100)	Income Cluster
0 Male 19 15 39 0

1 Male 21 15 81 0

2 Female 20 16 6 0

3 Female 23 16 77 0

4 Female 31 17 40 0

vi) Obtain the number of data values in each cluster.

Code: df['Income Cluster'].value_counts()

Output:

- The **'1' cluster** has the largest number of customers i.e **92**.

- The **'0' cluster** has the second largest number of customers i.e **72**.

- The **'2'** cluster has the least number of customers i.e **36**.
vii) Summary statistics of the clusters obtained.

The income cluster 2 has the highest level of annual income in addittion to the highest spending score.

Code: df.groupby('Income Cluster')['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()

Output:

                Age	Annual Income (k$)	Spending Score (1-100)

 Income Cluster

               0	38.930556	33.027778	50.166667

               1	39.184783	66.717391	50.054348

                2	37.833333	99.888889	50.638889
Bivariate Clustering.
i). Initiate the KMeans clustering algorithm,fit the data values,label the obtained clusters, create new column,obtain dataset overview.

Code:

clustering2 = KMeans(n_clusters=5)

clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])

df['Spending and Income Cluster'] =clustering2.labels_

df.head()

Output:

  Gender	       Age	    Annual Income (k$)	  Spending Score (1-100)	Income Cluster	Spending and Income Cluster

0	Male	        19	            15	                      39	               0	                       4

1	Male	        21	             15	                      81	                0	                       3

2	Female	      20	             16	                      6                  	0	                       4

3	Female	      23	             16                       77                 	0	                        3

4	Female	      31	             17                        40	                 0	                       4
ii). Obtain the appropriate number of clusters for the dataset using the elbow method.

From the plot,I was able to obtain that the suitable number of clusters is 5.

Code:

intertia_scores2=[]

for i in range(1,11):

kmeans2=KMeans(n_clusters=i)

kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])

intertia_scores2.append(kmeans2.inertia_)

plt.plot(range(1,11),intertia_scores2)

image

iii). Scatter plot to display the relationship between the 'Spending score' and 'Annual Income' in reference to clusters.

Code 1:

centers =pd.DataFrame(clustering2.cluster_centers_)

centers.columns = ['x','y']

Code 2:

plt.figure(figsize=(10,8))

plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')

sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')

plt.savefig('clustering_bivaraiate.png')

image

iv) Compare the male and female gender by the 'Spending and Income Cluster'.

Code: pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')

Output:

                Gender	     Female       	Male
Spending and Income Cluster

        0                       0.592593	      0.407407
     
        1	                     0.538462	      0.461538
      
        2	                     0.457143	      0.542857
      
        3	                     0.590909	      0.409091
      
        4	                     0.608696	      0.391304
v). Group the variables by the 'Spending and Income' Cluster.

Code: df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()

Output:

                                    Age	         Annual Income (k$)	      Spending Score (1-100)
Spending and Income Cluster

        0	                        42.716049	         55.296296	                49.518519

        1	                        32.692308	          86.538462                	82.128205

        2	                         41.114286	        88.200000	                 17.114286

        3	                         25.272727	         25.727273	                79.363636

        4	                         45.217391	         26.304348	                 20.913043
Reccommendations Obtained From The Project.
The aim of the project is to identify the best target customer group(s) in order for the marketing team to plan the best optimal strategy that ought to increase the mall's product sales.

Following my analysis above,my reccommendations are as follows:

1). The best target customer group is: Cluster 1.

This is because this cluster has both a high spending score and high income thus their able to purchase more products and at an often rate too.

54% of cluster 1 customers are women.The marketing team could run a marketing campaign advertising items that particularly resonate with this sub-cluster.

2) Cluster 3 also presents an interesting opportunity to increase product sales.

Although this cluster is in the low income range,it is suprisingly also in the high spending score category.

The customers could potentially be inclined to purchase more products when popular items are discounted.
