import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# 将字体设置成包含中文的字体，使matplotlib能够正常显示中文
plt.rc("font", family='YouYuan')

#读入csv文件
data=pd.read_csv('films.csv')
data.info()

#添加新的一列date，将日期的月日转化为整型类型，方便后期pca
def getdate(item):
    s=item.split('-')
    date=int(s[1])*100+int(s[2])
    return date

data['date']=data['日期'].map(getdate)


#【使用knn插补法填补'累计上映日期'的缺失值】
# 选择需要计算相似度的特征列
features_to_normalize = data[['排名','电影ID','累计上映天数','当前票房(万)','累计票房(万)','累计场次','累计人次(万)',
                              '票房占比','当前场次','当前人次(万)','人次占比','场均人次','场均收入','黄金场票房(万)',
                              '黄金场场次','黄金场人次(万)','黄金场排座(万)','黄金场场均人次','票房环比','场次环比',
                              '人次环比','场次占比','上午场票房(万)','上午场场次','上午场人次(万)','下午场票房(万)','下午场场次',
                              '下午场人次(万)','加映场票房(万)','加映场场次','加映场人次(万)','上座率','黄金场票房占比',
                              '黄金场场次占比','黄金场人次占比','黄金场上座率','票房占全国比','当前排座(万)','排座占比']]  # 选择需要计算相似度的特征列

# 使用MinMaxScaler方法对特征进行归一化处理
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features_to_normalize)
normalized_features_df = pd.DataFrame(normalized_features, columns=features_to_normalize.columns)

# 使用KNNImputer填充缺失值
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(normalized_features_df)

data_restored = scaler.inverse_transform(data_imputed)     #恢复归一化
data_restored_df = pd.DataFrame(data_restored, columns=features_to_normalize.columns)  #转化为dataframe类


# 将填充后的数据中的B列转化为整数类型
data_restored_df['累计上映天数'] = data_restored_df['累计上映天数'].astype(int)

#将原始的data的'累计上映天数'列用填充后的新列替换
data['累计上映天数']=data_restored_df['累计上映天数']

#打印输出到新的csv文件，方便后期直接读入进行操作
#data.to_csv('填充后数据.csv',index=False)

#【'累计上映日期'列缺失值填补完成】



#！！！【下列一些代码为数据预处理过程中用来检查数据一些特点的代码，由于后期需要一些预处理操作，故注释掉了，可根据需要去掉注释，运行相应代码】

# 计算特征的众数
#mode = data['累计上映天数'].mode()[0]

# 将日期特征转换为日期时间类型
#data['上映日期'] = pd.to_datetime(data['上映日期'])


#print(pd.DataFrame(data_filled[column_to_fill],))

# 按月份汇总数据
#monthly_data = data.groupby(data['上映日期'].dt.to_period('M')).size()

# 绘制直方图
#fig, ax = plt.subplots(figsize=(1000, 10))
#monthly_data.plot(kind='bar', color='skyblue', ax=ax)
#ax.set_title('日期特征按月份分布情况')
#ax.set_xlabel('月份')
#ax.set_ylabel('频数')
#plt.show()
# 用特征的众数填充缺失值
#data['特征名'].fillna(mode, inplace=True)

#data.info()      #可用来查看各特征的数据类型
#输出某一列所有数据：print(data.日期)
#print(data)

#print(data.duplicated().sum())   #查看重复数据数量

#print(data.isnull().sum()) #打印输出每一列缺失数据的和
#print(data[data.影片英文名称.isnull()])

#填补“无英文名称”
#data.loc[data.影片英文名称.isnull(),'影片英文名称']='无英文名称'
#print(data[data.上映日期.isnull()])
#print(data.isnull().sum())

#df=df.drop_duplicates()
#data=np.array(data)
#print(data)
#data_1=data.loc[:,['电影名称','天数']]
#data_1=data_1['电影名称'].value_counts()
#print(data_1)

#pd.DataFrame

import numpy as np

#selected_columns = data.iloc[:, 5:17]
# 检测异常值
#def detect_outliers(column):
 #   mean = column.mean()
  #  std_dev = column.std()
   # threshold = 3 * std_dev
    #outliers = data[(data[column.name] - mean).abs() > threshold]
    #return outliers

# 遍历选定的列，检测异常值并打印输出
#for column in selected_columns:
 #   outliers = detect_outliers(selected_columns[column])
  #  if not outliers.empty:
   #     print(column, "的异常值：\n", outliers[column])
    #    print("异常值所在样本的特征：\n", outliers['电影名称'])
