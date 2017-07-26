# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:17:23 2017

@author: SamLIN
"""
#---------------------------------------------------------------------------------
# 使用pandas库来读取.csv文件
import pandas as pd

# 创建pandas dataframe对象并赋值予变量titanic
titanic = pd.read_csv(r"C:\Users\SamLIN\Kaggle\Titanic_Machine Learning from Disaster\train.csv") 

# 输出dataframe的描述信息
print(titanic.describe())


# 数据清洗-------------------------------------------------------------------------
# 数据清洗Age缺失补齐为中位数；Sex转换为0~2；Embarked缺失补齐为S，转换为0~2
# 调用.median()属性获取中位数
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# 确认所有不重复值——应该只有male/female
print(titanic["Sex"].unique())

# 将male替换为1,将female替换为0
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1 
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

# 输出"Embarked"的所有数据
print(titanic["Embarked"].unique())

# 首先把所有缺失值替换为"S"
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# 将"S"替换为0,将"C"替换为1,将"Q"替换为2
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 0

# 提取新特征=======================================================================
# 生成新特征列：家庭规模------------------------------------------------------------
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# 用.apply()方法生成新特征列：姓名长度
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))


# 生成新特征列：称谓---------------------------------------------------------------
# 正则化库
import re

# 从姓名中提取称谓[‘Master.’,‘Mr.’,‘Mrs.’等]的函数
def get_title(name):
    # 正则表达式检索称谓，称谓总以大写字母开头并以句点结尾
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果称谓存在则返回其值
    if title_search:
        return title_search.group(1)
    return ""

# 创建一个新的Series对象titles，统计各个头衔出现的频次
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))

# 将每个称谓映射到一个整数，有些太少见的称谓可以压缩到一个数值
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# 验证转换结果
print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles


# 生成新特征列：家庭组成-----------------------------------------------------------
import operator

# 映射姓氏到家庭ID的字典
family_id_mapping = {}

# 从行信息提取家庭ID的函数
def get_family_id(row):
    # 分割逗号获取姓氏
    last_name = row["Name"].split(",")[0]
    # 创建家庭ID列表
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # 从映射中查询ID
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # 遇到新的家庭则将其ID设为当前最大ID+1
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# 用.apply()方法获得家庭ID
family_ids = titanic.apply(get_family_id, axis=1)

# 家庭数量过多，所以将所有人数小于3的家庭压缩成一类
family_ids[titanic["FamilySize"] < 3] = -1

# 输出每个家庭ID的数量
print(pd.value_counts(family_ids))

titanic["FamilyId"] = family_ids

# 最佳特征------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# 特征选择
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# 得到每个特征列的p值，再转换为交叉验证得分
scores = -np.log10(selector.pvalues_)

# 绘制得分图像，观察哪个特征是坠好的
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# 只选取最好的四个特征
predictors = ["Pclass", "Sex", "Fare", "Title"]




# 计算============================================================================
# 计算方案1:逻辑回归 --------------------------------------------------------------
## 导入逻辑回归类和交叉验证
#from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation
#
## 定义纳入训练过程的数据列
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#
## 初始化逻辑回归模型对象
#alg = LogisticRegression(random_state=1)
#
## 计算交叉验证的得分
#scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=10)
#
## 求scores的平均值并输出
#print(scores.mean())

# 计算方案2:随机森林 --------------------------------------------------------------
## 导入随机森林分类器
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import cross_validation
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
## n_estimators：决策树的数量
## min_samples_split：单个分叉包含的最小行数
## min_samples_leaf：单个叶节点的最小样本数量
#alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
#
## 计算交叉验证的得分
#kf = cross_validation.KFold(titanic.shape[0], n_folds=10, random_state=1)
#scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
#
## 输出得分的平均值
#print(scores.mean())


# 计算方案3:随机森林advanced --------------------------------------------------------------
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

alg2 = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# 计算交叉验证得分
scores = cross_validation.cross_val_score(alg2, titanic[predictors], titanic["Survived"], cv=3)


# 输出交叉验证得分的均值
print(scores.mean())










## 测试============================================================================
#titanic_test = pd.read_csv(r"C:\Users\SamLIN\Kaggle\Titanic_Machine Learning from Disaster\test.csv")
#
## 用中位数替换"Age"的缺失数据
#titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
#
## 用中位数替换"Fare"的缺失数据
#titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#
## 将"Sex"和"Embarked"数值化
#titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1 
#titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0
#
#titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
#titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
#titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
#titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
#
#
#
## 训练算法
#alg.fit(titanic[predictors], titanic["Survived"])
#
## 对测试集做预测
#predictions = alg.predict(titanic_test[predictors])
#
## 创建新的dataframe对象submission，仅含"PassengerID"和"Survived"两列。
#submission = pd.DataFrame({
#        "PassengerId": titanic_test["PassengerId"],
#        "Survived": predictions
#    })
#
## 将submission输出为.csv格式的提交文件
#submission.to_csv("kaggle2.csv", index=False)

