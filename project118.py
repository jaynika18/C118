import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sb
import matplotlib.pyplot as mlt


df = pd.read_csv("project118.csv")


fig = px.scatter(df , x = 'Size', y = 'Light', title = "Graph to study the data")
# fig.show()


X = df.iloc[: , [0,1]].values
# print(X)

WCSS = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i ,init="k-means++" , random_state=42)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

mlt.figure(figsize=(10,5))
sb.lineplot( range(1,11), WCSS, marker='o' , color="cyan")   
mlt.title("Elbow Graph")
mlt.xlabel("Number of Clusters")
mlt.ylabel("WCSS")
# mlt.show()


kmeans = KMeans(n_clusters=3 ,init="k-means++" , random_state=42)
y_kmeans = kmeans.fit_predict(X) 

mlt.figure(figsize=(15,7))
sb.scatterplot( X[y_kmeans == 0,0] , X[y_kmeans == 0,1] , color = "cyan" , label = "Cluster 1" )
sb.scatterplot( X[y_kmeans == 1,0] , X[y_kmeans == 1,1] , color = "green" , label = "Cluster 2" )
sb.scatterplot( X[y_kmeans == 2,0] , X[y_kmeans == 2,1] , color = "red" , label = "Cluster 3" )

sb.scatterplot(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[:,1] , color="yellow" , label="Centroids" , s=100)

mlt.grid(False)
mlt.title("Cluster graph")
mlt.xlabel("Size")
mlt.ylabel("Light")
mlt.legend()
mlt.show()