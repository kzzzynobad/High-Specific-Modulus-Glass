from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from itertools import product


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

# 定义隐藏层和神经元选择范围
hidden_layers = [1, 2, 3]  # 可选隐藏层数
neurons_range = range(1, 51, 1)  # 每层神经元数范围，从50到200，步长为50

# 生成所有可能的隐藏层和神经元组合的列表
hidden_layer_choices = []
for num_layers in hidden_layers:
    for combination in product(neurons_range, repeat=num_layers):
        hidden_layer_choices.append(combination)
print(hidden_layer_choices[125888])
act = ['identity', 'logistic', 'tanh', 'relu']
space = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', hidden_layer_choices),
    'alpha': hp.uniform('alpha', 0.0001, 1.0),
    'learning_rate_init': hp.uniform('learning_rate_init', 0.0001, 0.1),
    'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
}


def objective(params):
    clf = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'],
                       alpha=params['alpha'], learning_rate_init=params['learning_rate_init'],
                       activation=params['activation'], random_state=42, solver='adam')
    score = -cross_val_score(clf, x, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score


# 运行贝叶斯优化搜索
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print("最佳参数组合：", best)
# 示例模型
model = MLPRegressor(hidden_layer_sizes=hidden_layer_choices[best['hidden_layer_sizes']],
                     alpha=best['alpha'], learning_rate_init=best['learning_rate_init'],
                     activation=act[best['activation']], random_state=42, solver='adam')
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