import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
def main():

    # 加载数据
    try:
        TCSC = pd.read_csv('data2.csv')
    except FileNotFoundError:
        print("错误：无法找到文件 'data.csv'")
        return

    # 准备数据
    X = TCSC.drop('Result', axis=1)
    y = TCSC['Result']

    SS = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    X_train = SS.fit_transform(X_train)

    # 初始模型 SVM
    svc = SVC(kernel='rbf', decision_function_shape='ovo')
    svc.fit(X_train, y_train)
    predict = svc.predict(SS.transform(X_test))
    print("初始模型分类报告:")
    print(classification_report(y_test, predict))

    # 网格搜索优化
    param_grid = {'C': np.logspace(-3, 5, 70), 'gamma': np.logspace(-3, 5, 70)}
    grid = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovo'), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    predict_grid = grid.predict(SS.transform(X_test))
    print("网格搜索后模型分类报告:")
    print(classification_report(y_test, predict_grid))

    # 决策边界图绘制
    scaled_X = SS.transform(X.to_numpy())
    x_min, x_max = scaled_X[:, 0].min() - 0.1, scaled_X[:, 0].max() + 0.1
    y_min, y_max = scaled_X[:, 1].min() - 0.1, scaled_X[:, 1].max() + 0.15
    h = 0.002
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = grid.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.flatten()

    # 映射分类值以增强可视化效果（可自定义）
    for i in range(len(Z)):
        if Z[i] == 0:
            Z[i] = -1
        elif Z[i] == 1:
            Z[i] = 1.5
        elif Z[i] == 2:
            Z[i] = 2.5
    Z = Z.reshape(xx.shape)

    # 自定义颜色
    contour_colors = ['#9ee6a0', '#ffffb6','#b6b6ff']  # 紫、绿、黄
    # contour_colors = ['#cee7a9', '#aaede9','#fbd0d0']  
    custom_cmap = ListedColormap(contour_colors)

    # 网格坐标转换为原始尺度
    xx_t = xx * X.iloc[:, 0].std() + X.iloc[:, 0].mean()
    yy_t = yy * X.iloc[:, 1].std() + X.iloc[:, 1].mean()

    # 图像绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx_t, yy_t, Z, levels=20, cmap=custom_cmap, alpha=0.7)
    cbar = fig.colorbar(contour, ax=ax)
    # cbar.set_label('预测分类值', fontsize=12)

    # 绘制原始数据点（可选）
    # scatter = ax.scatter(
    #     X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=custom_cmap,
    #     edgecolors='k', s=50, marker='o', alpha=0.8, label='训练样本'
    # )

    # 轴
    ax.set_xlabel('BA ration', fontsize=14, labelpad=10)
    ax.set_ylabel('Temperature (°C)', fontsize=14, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # ax.set_title('支持向量机分类决策边界图', fontsize=16, weight='bold', pad=15)
    # ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    ax.set_xlim(0, 1)  # x 坐标限制为 0 到 1
    ax.set_ylim(30, 50)  # y 坐标限制为 30 到 50

    plt.savefig('decision_boundary_2st.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()