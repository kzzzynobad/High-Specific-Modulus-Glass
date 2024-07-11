import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from hyperopt import hp, fmin, tpe, Trials


def objective(params):
    clf = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42)
    score = -cross_val_score(clf, x_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score


def score(pred, ture):
    MSE = mean_squared_error(pred, ture)
    MAE = mean_absolute_error(pred, ture)
    MedAE = median_absolute_error(pred, ture)
    R2 = r2_score(pred, ture)
    return MSE, MAE, MedAE, R2


# 读取CSV文件
data = pd.read_csv(r'C:\Users\15198\Desktop\ML1\tl\data2.csv')
# 提取特征和目标变量
x = data.drop(columns=['Tliquidus'])  # 特征
y = data['Tliquidus']  # 目标变量
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 定义超参数空间
space = {
    'n_estimators': hp.randint('n_estimators', 10, 501),
    'max_depth': hp.randint('max_depth', 1, 51),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1.0),
    'min_child_weight': hp.randint('min_child_weight', 1, 11),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}
# 运行贝叶斯优化搜索
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print("最佳参数组合：", best)
# 示例模型
model = xgb.XGBRegressor(**best, random_state=42, n_jobs=-1)
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