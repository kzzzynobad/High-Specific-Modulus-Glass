from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, fmin, tpe, Trials


def objective(params):
    clf = DecisionTreeRegressor(**params, random_state=42)
    score = -cross_val_score(clf, x_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score


def score(pred, ture):
    MSE = mean_squared_error(pred, ture)
    MAE = mean_absolute_error(pred, ture)
    MedAE = median_absolute_error(pred, ture)
    R2 = r2_score(pred, ture)
    return MSE, MAE, MedAE, R2


# 读取CSV文件
data = pd.read_csv(r'D:\高比模量\ML1\den\data2.csv')
# 提取特征和目标变量
x = data.drop(columns=['Density293K'])  # 特征
y = data['Density293K']  # 目标变量

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 定义超参数空间
space = {
    'max_depth': hp.randint('max_depth', 1, 51),
    'min_samples_split': hp.randint('min_samples_split', 2, 11),
    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 11),
}
# 运行贝叶斯优化搜索
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print("最佳参数组合：", best)
# 示例模型
model = DecisionTreeRegressor(max_depth=best['max_depth'], min_samples_split=best['min_samples_split'], min_samples_leaf=best['min_samples_leaf'], random_state=42)
R2 = cross_val_score(model, x_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
MSE = -cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
MAE = -cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
MedAE = -cross_val_score(model, x_train, y_train, cv=5, scoring='neg_median_absolute_error', n_jobs=-1).mean()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train = score(y_train_pred, y_train)
test = score(y_test_pred, y_test)
print(R2)
print(MSE)
print(MAE)
print(MedAE)
print('Train')
print(train[3])
print(train[0])
print(train[1])
print(train[2])
print('Test')
print(test[3])
print(test[0])
print(test[1])
print(test[2])



