import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
# 逻辑回归模型

from sklearn.linear_model import LogisticRegression
#用来拆分训练和测试数据
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

#读取数据
data = pd.read_csv('/home/anaconda/data/banking.csv',header=0)
data = data.dropna()
#打印数据大小
print(data.shape)
#打印数据的列名

print(list(data.columns))

#计算正样本和负样本的比例
count_no_sub = len(data[data['y']==0])#计算负样本个数，len()返回列表元素个数
count_sub = len(data[data['y']==1])#计算正样本个数
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)#计算百分比
print('未开户的百分比:  %.2f%%' % (pct_of_no_sub*100))
pct_of_sub = count_sub/(count_no_sub+count_sub)
print('开户的百分比：%.2f%%' % (pct_of_sub*100))

#把“education”字段里的三个值“basic.9y”，“basic.6y”，“basic.4y”合并成同一个值“Basic”
print(data['education'].unique())

#np.where()用法：np.where(condition, x, y)，condition中需要包含一个待处理的ndarray(这里记为A)，那么对于A中的每个元素，如果满足条件，则将这个元素替换为x，否则，将这个元素替换为y。
data['education']=np.where(data['education']=='basic.9y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.6y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.4y','Basic',data['education'])

#数据可视化分析：JOB字段和预测变量之间关系

table=pd.crosstab(data.job,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked bar Chart of job title vs Purchase')
plt.ylabel('job')
plt.ylabel('Proportion of Purchase')
plt.savefig('purchase_vs_job')

#数据可视化分析：Marital Status字段和预测变量之间关系
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Marital Starus vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')


#数据可视化分析：Education字段和预测变量之间关系
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')

#数据可视化分析：Day of week字段和预测变量之间关系
#我们可以发现，这个字段和预测变量之间关系相对较弱
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Day of week vs Purchase')
plt.xlabel('Day of week')
plt.ylabel('Proportion of Purchase')
plt.savefig('dow_vs_pur_stack')

#类别型变量需要转换成独热编码形式，列出所有类型变量
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    #TODO每个变量转换成类别型变量，参考函数 pd.get_dummies
    cat_list=pd.get_dummies(data[var],prefix=var)
    data=data.join(cat_list)

#剔除掉原来类别型变量，只保留独热编吗
data_final=data.drop(cat_vars,axis=1)
data_final.columns.values


#TODO构造训练数据，X为特征，Y为标签
X=data_final.loc[:,data_final.columns !='y']
y=data_final.loc[:,data_final.columns == 'y'].values.ravel()

#TODO把数据分为训练和测试数据
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#TODO初始化逻辑回归模型并在训练数据上训练
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#TODO计算F1-SCORE，使用classification_report函数
y_pred=logreg.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))




