import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


data=pd.read_csv('有标签数据.csv')


# 对数据集进行拆分，80% 用于训练，20% 用于测试
# 特征
X=data.loc[:,['date','排名','电影ID','天数','累计上映天数','当前票房(万)','累计票房(万)','累计场次','累计人次(万)',
                              '票房占比','当前场次','当前人次(万)','人次占比','场均人次','场均收入','黄金场票房(万)',
                              '黄金场场次','黄金场人次(万)','黄金场排座(万)','黄金场场均人次','票房环比','场次环比',
                              '人次环比','场次占比','上午场票房(万)','上午场场次','上午场人次(万)','下午场票房(万)','下午场场次',
                              '下午场人次(万)','加映场票房(万)','加映场场次','加映场人次(万)','上座率','黄金场票房占比',
                              '黄金场场次占比','黄金场人次占比','黄金场上座率','票房占全国比','当前排座(万)','排座占比']]

y = data['是否为爆款电影']  # 目标变量

'''
# 使用 SMOTE 进行过采样
smote = SMOTE(sampling_strategy='auto')
X_smote, y_smote = smote.fit_resample(X, y)

# 拆分过采样后的数据集
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier()

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 打印分类报告和混淆矩阵
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# 打印分类报告
report = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
plt.title('Classification Report')
plt.show()

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


'''


# 使用SMOTE方法处理样本不平衡问题
oversampler = SMOTE(random_state=42)
X_res, y_res = oversampler.fit_resample(X, y)

# 创建随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 使用10折交叉验证进行模型评估
scores = cross_val_score(rf, X_res, y_res, cv=StratifiedKFold(n_splits=10), scoring='accuracy')

# 打印交叉验证准确率
print('交叉验证准确率:', scores)

# 存储交叉验证的准确率
scores = cross_val_score(rf, X_res, y_res, cv=StratifiedKFold(n_splits=10), scoring='accuracy')

# 绘制准确率折线图
plt.figure()
plt.plot(range(1, 11), scores)
plt.title('10-Fold Cross-Validation Accuracy')
plt.xlabel('Number of Folds')
plt.ylabel('Accuracy')
plt.show()
