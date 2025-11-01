import mysql.connector
import traceback
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


class Connector:
    def __init__(self,server="localhost", port=3306, database="salesdatabase", username="root", password="@Obama123"):
        self.server=server
        self.port=port
        self.database=database
        self.username=username
        self.password=password
    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.server,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                use_pure=True)
            return self.conn
        except:
            self.conn=None
            traceback.print_exc()
        return None

    def disConnect(self):
        if self.conn != None:
            self.conn.close()

    def queryDataset(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df.columns=cursor.column_names
            return df
        except:
            traceback.print_exc()
        return None
    def getTablesName(self):
        cursor = self.conn.cursor()
        cursor.execute("Show tables;")
        results=cursor.fetchall()
        tablesName=[]
        for item in results:
            tablesName.append([tableName for tableName in item][0])
        return tablesName
    def fetchone(self,sql,val):
        cursor = self.conn.cursor()
        cursor.execute(sql, val)
        one_item = cursor.fetchone()
        cursor.close()
        return one_item
    def fetchall(self,sql,val):
        cursor = self.conn.cursor()
        cursor.execute(sql, val)
        items = cursor.fetchall()
        cursor.close()
        return items
    def savedata(self,sql,val):
        cursor = self.conn.cursor()
        cursor.execute(sql, val)
        self.conn.commit()
        result=cursor.rowcount
        cursor.close()
        return result
def showHistogram(df, columns):
    plt.figure(1, figsize=(7, 8))
    n = 0
    for column in columns:
        n += 1
        plt.subplot(3, 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32, kde=True)  # sns.histplot thay vì distplot (distplot bị deprecated)
        plt.title(f'Histogram of {column}')
    plt.show()

def elbowMethod(df, columnsForElbow):
    X=df.loc[:,columnsForElbow].values
    inertia = []

    for n in range(1, 11):
        model = KMeans(
            n_clusters=n,
            init='k-means++',
            max_iter=500,
            random_state=42
        )
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(1, figsize=(15, 6))
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster sum of squared distances')
    plt.title('Elbow Method for Optimal K')
    plt.show()


conn=Connector(database="salesdatabase")
conn.connect()
sql="select * from customer"
df=conn.queryDataset(sql)
print(df)

sql2=("select distinct customer.CustomerId, Age, Annual_Income,Spending_Score from customer, customer_spend_score "
      "where customer.CustomerId=customer_spend_score.CustomerID")
df2=conn.queryDataset(sql2)
print(df2)

showHistogram(df2, df2.columns[1:])

columns=['Age','Spending_Score']
elbowMethod(df2,columns)

def runKMeans(X, cluster):
    model = KMeans(
        n_clusters=cluster,
        init='k-means++',
        max_iter=500,
        random_state=42
    )

    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)

    return y_kmeans, centroids, labels


# Giả sử df2 đã có dữ liệu
# df2 = ...  # Dữ liệu đã đọc từ SQL (bạn đã có phần này trước rồi)

columns = ['Age', 'Spending_Score']
X = df2.loc[:, columns].values
cluster = 4
colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]

# Chạy mô hình KMeans
y_kmeans, centroids, labels = runKMeans(X, cluster)

# In kết quả
print("📊 Nhãn cụm (y_kmeans):")
print(y_kmeans)
print("\n📍 Tọa độ tâm cụm (centroids):")
print(centroids)
print("\n🏷️ Nhãn gán cho từng điểm (labels):")
print(labels)

# Gắn nhãn cluster vào DataFrame
df2["Cluster"] = labels
print("\n🔹 DataFrame sau khi thêm cột Cluster:")
print(df2.head())

def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(X[y_kmeans == i, 0],
                    X[y_kmeans == i, 1],
                    s=100,
                    c=colors[i],
                    label='Cluster %i' % (i + 1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

visualizeKMeans(
    X,
    y_kmeans,
    cluster,
    "Clusters of Customers - Age X Spending Score",
    "Age",
    "Spending_Score",
    colors
)

X = df2.loc[:, columns].values
cluster =5
y_kmeans,centroids,labels=runKMeans(X,cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2['cluster']=labels
visualizeKMeans(X,y_kmeans,cluster,"Clusters of Customers - Annual Income X Spending Score",
                "Annual_Income","Spending_Score",colors)



columns = ['Age', 'Annual_Income', 'Spending_Score']
elbowMethod(df2, columns)
X = df2.loc[:, columns].values
cluster = 6

y_kmeans, centroids, labels = runKMeans(X, cluster)

print(y_kmeans)
print(centroids)
print(labels)

df2["cluster"] = labels
print(df2)
def visualize3DKmeans(df, columns, hover_data, cluster):
    fig = px.scatter_3d(df,
                        x=columns[0],
                        y=columns[1],
                        z=columns[2],
                        color='cluster',
                        hover_data=hover_data,
                        category_orders={"cluster": range(0, cluster)},
                        )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
hover_data = df2.columns
visualize3DKmeans(df2, columns, hover_data, cluster)
