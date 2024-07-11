import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from IPython.display import display
import shap
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt


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
# 示例模型
model = xgb.XGBRegressor(n_estimators=453, max_depth=15, learning_rate=0.133872361,
                         min_child_weight=1, gamma=0.24928596, reg_alpha=0.838069133,
                         reg_lambda=0.995660665, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train = score(y_train_pred, y_train)
test = score(y_test_pred, y_test)
print('Train')
print('MSE:', train[0])
print('MAE:', train[1])
print('MedAE:', train[2])
print('R2:', train[3])
print('Test')
print('MSE:', test[0])
print('MAE:', test[1])
print('MedAE:', test[2])
print('R2:', test[3])
# shap.initjs()
# 读取CSV文件
data1 = pd.read_csv(r'D:\高比模量\ML1\生成数据集\pred.csv')
# 提取特征和目标变量
x_pred = data1.iloc[0:, 0:17]  # 特征
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(x_pred)
# explanation = explainer(x_test)
# shap.summary_plot(shap_values, x_train, plot_type="bar")
# shap.summary_plot(shap_values, x_train)
explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_pred)
# shap.summary_plot(shap_values, x_test, plot_type="bar")
# # 获取当前的 matplotlib 图表对象
# fig = plt.gcf()
# # 自定义两种颜色交替使用
# bar_colors = ['#F2766D', '#00BFC4']  # 自定义两种颜色
# # 获取柱子列表
# bars = fig.gca().patches
# # 循环设置柱子颜色
# for idx, bar in enumerate(bars):
#     color = bar_colors[idx % len(bar_colors)]  # 使用取余来交替使用颜色
#     bar.set_color(color)
# # 显示图表
# plt.show()
# shap.summary_plot(shap_values, x_test)
# shap.plots.force(shap_values_sampled[0])
shap.plots.waterfall(shap_values[0], max_display=20, show=False)
fig = plt.gcf()  # 获取当前的图形对象

# 使用 Matplotlib 进行进一步的定制
ax = fig.get_axes()[0]  # 获取瀑布图的轴对象

# # 修改 x 轴刻度的示例
# ax.set_xticks(np.arange(1400, 1625, 25))  # 设置刻度在每个数据点处
ax.set_xlim(1435, 1625)
# # 修改 y 轴刻度的示例
# ax.set_yticks(np.arange(1400, 1650, 100))  # 设置自定义的 y 轴刻度

plt.show()
# shap_plot = shap.plots.force(explanation)
# shap.save_html(r'D:\高比模量\shap_plot.html', shap_plot)
# pl.savefig('force.png', bbox_inches='tight', dpi=600)
# for name in x_train.columns:
#     shap.dependence_plot(name, shap_values, x_test, display_features=x, interaction_index=None, show=False)
#     plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
#     plt.xlim(0, 100)
#     plt.show()