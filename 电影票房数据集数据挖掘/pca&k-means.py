import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# 将字体设置成包含中文的字体，使matplotlib能够正常显示中文
plt.rc("font", family='YouYuan')

# 读取CSV文件
data = pd.read_csv('填充后数据.csv')

# 使用to_datetime函数将日期字符串转换为日期类型
#data['日期'] = pd.to_datetime(data['日期'])


#【对数据进行PCA降维】
data_=data.loc[:,['date','排名','电影ID','天数','累计上映天数','当前票房(万)','累计票房(万)','累计场次','累计人次(万)',
                              '票房占比','当前场次','当前人次(万)','人次占比','场均人次','场均收入','黄金场票房(万)',
                              '黄金场场次','黄金场人次(万)','黄金场排座(万)','黄金场场均人次','票房环比','场次环比',
                              '人次环比','场次占比','上午场票房(万)','上午场场次','上午场人次(万)','下午场票房(万)','下午场场次',
                              '下午场人次(万)','加映场票房(万)','加映场场次','加映场人次(万)','上座率','黄金场票房占比',
                              '黄金场场次占比','黄金场人次占比','黄金场上座率','票房占全国比','当前排座(万)','排座占比']]

# 使用StandardScaler进行标准化数据归一化
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_)

# 定义PCA对象，设置主成分数目（例如，n_components=2）
pca = PCA(n_components=10)  # ！可以根据需要更改n_components的值

# 对归一化的数据进行PCA降维
pca_data = pca.fit_transform(normalized_data)

# 将降维后的数据转换回DataFrame
pca_data_df = pd.DataFrame(data=pca_data)

#将降维结果保存到了一个csv文件中【因后期操作注释掉了，可去掉注释运行该行代码】
#pca_data_df.to_csv('pca_result.csv',index=False)

'''
# 【查看各个特征值的贡献率并可视化】（可去掉注释后运行查看）
evr = pca.explained_variance_ratio_
plt.bar(range(len(evr)), evr, color='b', alpha=0.5)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.show()
print("特征贡献率:")
print(pca.explained_variance_ratio_)
'''
#【PCA降维结束】



#【使用K-means进行聚类】
X=pca_data_df
'''
#使用轮廓系数来选择最优的k值
scores=[]     # 用于存储每个K值下的聚类性能得分
for i in range(2,18): # 可能的K值范围设置为2-18
    km=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)# 创建一个KMeans对象
    km.fit(X)
    scores.append(metrics.silhouette_score(X,km.labels_,metric='euclidean'))  # 计算聚类性能得分，放入到scores中
    
# 绘制轮廓系数曲线图
plt.figure(dpi=150)
plt.plot(range(2, 18),scores,marker='o')
plt.xlabel='K值'
plt.ylabel='轮廓系数'
plt.title('不同K值下的聚类性能')
plt.s
'''

#经过分析后，选取k=2进行聚类
km=KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,random_state=0)  # 创建一个KMeans对象
km.fit(X)


data['cluster_label'] = km.labels_
data = data.rename(columns={'cluster_label': '是否为爆款电影'})
#print(data.info())


#统计爆款电影和非爆款电影数量
x1=(data['是否为爆款电影']==1).sum()
x2=(data['是否为爆款电影']==0).sum()
print('爆款电影数量：',x1)
print('非爆款电影数量：',x2)


#将有标签的数据保存，方便查看聚类结果(可根据需要去掉注释)
#data.to_csv('有标签数据.csv',index=False)


#【展示聚类前后的tsne图】(因为该步骤运行起来浪费时间，故在查看tsne图后注释了起来，可根据需要去掉注释运行)
'''
# 对聚类前的数据进行t-SNE降维
tsne_before = TSNE(n_components=2, random_state=0).fit_transform(data_)

# 对聚类后的数据进行t-SNE降维
tsne_after = TSNE(n_components=2, random_state=0).fit_transform(pca_data_df)

# 可视化聚类前的数据
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(tsne_before[:, 0], tsne_before[:, 1], c='b', label='Non-clustered')
plt.title('t-SNE Plot of Data before Clustering')

# 可视化聚类后的数据
plt.subplot(1, 2, 2)
plt.scatter(tsne_after[:, 0], tsne_after[:, 1], c=km.labels_, cmap='viridis', label='Clustered')
plt.title('t-SNE Plot of Data after Clustering')

plt.show()
'''