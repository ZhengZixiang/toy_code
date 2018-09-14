# -*- coding: utf-8 -*-
# 自行车租赁数据分析与可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd  # 读取数据到DataFrame
import numpy as np
import urllib  # 获取网咯数据
import tempfile  # 创建临时文件系统
import shutil  # 文件操作
import zipfile  # 压缩解压
import statsmodels.api as sm # 最小二乘
from statsmodels.stats.outliers_influence import summary_table  # 获得汇总信息
from scipy.stats import gaussian_kde


# step 1 导入数据，做简单的数据处理
temp_dir = tempfile.mkdtemp()  # 建立临时目录
data_source = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'  # 网络数据地址
zipname = temp_dir + '/Bike-sharing-Dataset.zip'  # 拼接文件和路径
urllib.request.urlretrieve(data_source, zipname)  # 获得数据

zip_ref = zipfile.ZipFile(zipname, 'r')  # 创建一个ZipFile对象处理压缩文件
zip_ref.extractall(temp_dir)  # 解压
zip_ref.close()

daily_path = '/day.csv'
daily_data = pd.read_csv(temp_dir + daily_path)  # 读取csv文件
daily_data['dteday'] = pd.to_datetime(daily_data['dteday'])  # 把字符串数据转换成日期数据
drop_list = ['instant', 'season', 'yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'atemp', 'hum']  # 不关注的列
daily_data.drop(drop_list, inplace=True, axis=1)  # inplace=True 在对象上直接操作

shutil.rmtree(temp_dir)  # 接触临时文件目录


# step 2 配置参数
# 设置一些全局的资源参数，可以进行个性化修改 rc resource configuration
# 设置图片尺寸 14 x 7
mpl.rc('figure', figsize=(14, 7))
# 设置字体 14
mpl.rc('font', size=14)
# 不显示顶部和右侧的坐标线
mpl.rc('axes.spines', top=False, right=False)
# 显示网格
mpl.rc('axes', grid=True)
# 设置背景颜色是白色
mpl.rc('axes', facecolor='white')


# step 3 关联分析
# 散点图 分析变量关系
# 包装一个散点图的函数便于复用
def scatterplot(x_data, y_data, x_label, y_label, title):

    # 创建一个绘图对象
    fig, ax = plt.subplots()

    # 设置数据、点的大小、点的颜色和透明度
    ax.scatter(x_data, y_data, s=10, color='#539caf', alpha=0.74)

    # 添加标题和坐标说明
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 绘制散点图
scatterplot(x_data=daily_data['temp'],
            y_data=daily_data['cnt'],
            x_label='Normalized Temperature (C)',
            y_label='Check Outs',
            title='Number of Check Outs vs Temperature')
plt.show()


# 曲线图 拟合变量关系
# 包装曲线绘制函数
def lineplot(x_data, y_data, x_label, y_label, title):
    # 创建一个绘图对象
    _, ax = plt.subplots()

    # 绘制拟合曲线，lw=linewidth, alpha=transparancy
    ax.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)

    # 添加标题和坐标说明
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 线性回归
x = sm.add_constant(daily_data['temp'])  # 线性回归增加常数项 y=kx+b
y = daily_data['cnt']
regression = sm.OLS(y, x)  # 普通最小二乘模型 ordinary least square model
result = regression.fit()
# 从模型获得拟合数据
st, data, ss2 = summary_table(result, alpha=0.05)  # 置信水平alpha=5%，st数据汇总，data数据详情，ss2数据列名
fitted_values = data[:, 2]  # 2拟合数据列

# 调用绘图函数
lineplot(x_data=daily_data['temp'],
         y_data=fitted_values,
         x_label='Normalized Temperature (C)',
         y_label='Check Outs',
         title='Line of Best Fit for Number of Check Outs vs Temperature')
plt.show()

# x.head()
# type(regr)
# st


# 带置信区间的曲线图 评估曲线拟合结果
# 包装置信区间曲线图绘制函数
def lineplot_ci(x_data, y_data, sorted_x, low_ci, upper_ci, x_label, y_label, title):

    # 创建一个绘图对象
    _, ax = plt.subplots()

    # 绘制预测曲线
    ax.plot(x_data, y_data, lw=1, color='#539caf', alpha=1, label='Fit')
    # 绘制置信区间，顺序填充
    ax.fill_between(sorted_x, low_ci, upper_ci, color='#539caf', alpha=0.4, label='95%CI')

    # 添加标题和坐标说明
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # 显示图例，配合label参数，loc='best'自适应方式
    ax.legend(loc='best')


# 获得5%置信区间的上下界
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T  # 4上界5下界

# 创建置信区间DataFrame，上下界
CI_df = pd.DataFrame(columns=['x_data', 'low_ci', 'upper_ci'])
CI_df['x_data'] = daily_data['temp']
CI_df['low_ci'] = predict_mean_ci_low
CI_df['upper_ci'] = predict_mean_ci_upp
CI_df.sort_values('x_data', inplace=True)  # 根据x_data进行排序

# Call the function to create plot
lineplot_ci(x_data=daily_data['temp'],
            y_data=fitted_values,
            sorted_x=CI_df['x_data'],
            low_ci=CI_df['low_ci'],
            upper_ci=CI_df['upper_ci'],
            x_label='Normalized Temperature (C)',
            y_label='Check Outs',
            title='Line of Best Fit for Number of Check Outs vs Temperature')
plt.show()


# 双坐标曲线图
# - 曲线拟合不满足置信阈值时，考虑增加独立变量
# - 分析不同尺度多变量的关系
def lineplot2y(x_data, x_label, y1_data, y1_label, y1_color,  y2_data, y2_label, y2_color, title):

    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color=y1_color)
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color=y1_color)

    ax2 = ax1.twinx()  # 两个绘图对象共享横坐标
    ax2.plot(x_data, y2_data, color=y2_color)
    ax2.set_ylabel(y2_label, color=y2_color)
    # 右侧坐标可见
    ax2.spines['right'].set_visible(True)


# 调用绘图函数
lineplot2y(x_data=daily_data['dteday'],
           x_label='Day',
           y1_data=daily_data['cnt'],
           y1_label='Check Outs',
           y1_color='#539caf',
           y2_data=daily_data['windspeed'],
           y2_label='Normalized Windspeed',
           y2_color='#7663b0',
           title='Check Outs and Windspeed Over Time')
plt.show()


# step 4 分布分析
# 灰度图 粗略区间计数
# 绘制灰度图函数
def histogram(data, x_label, y_label, title):
    _, ax = plt.subplots()
    res = ax.hist(data, color='#539caf', bins=10)  # 设置bin的数量
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return res


# 绘图函数调用
result = histogram(data=daily_data['registered'],
                   x_label='Check Outs',
                   y_label='Frequency',
                   title='Distribution of Registered Check Outs')
plt.show()
# print(result[0])
# print(result[1])


# 堆叠直方图 比较两个分布
# 绘制堆叠的直方图
def overlaid_histogram(data1, data1_name, data1_color, data2, data2_name, data2_color, x_label, y_label, title):
    # 归一化数据区间，对齐两个直方图的bins
    max_bins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    bin_width = (data_range[1] - data_range[0]) / max_bins
    bins = np.arange(data_range[0], data_range[1]+bin_width, bin_width)  # 生成直方图bins区间

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins=bins, color=data1_color, alpha=1, label=data1_name)
    ax.hist(data2, bins=bins, color=data2_color, alpha=0.4, label=data2_name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best')


# Call the function to create plot
overlaid_histogram(data1=daily_data['registered'],
                   data1_name='Registered',
                   data1_color='#539caf',
                   data2=daily_data['casual'],
                   data2_name='Casual',
                   data2_color='#7663b0',
                   x_label='Check Outs',
                   y_label='Frequency',
                   title='Distribution of Check Outs By Type')
plt.show()


# 密度图 精细可化概率分布
# KDE: kernal density estimate
# $\hat{f}_h(x)=\frac{1}{n}\sum_{i=1}^nK_h(x-x_i)=\frac{1}{nh}\sum_{i=1}^{n}K(\frac{x-x_i}{h})$
# 计算概率密度
# from scipy.stats import gaussian_kde
data = daily_data['registered']
density_est = gaussian_kde(data)  # kernal density estimate: https://en.wikipedia.org/wiki/Kernel_density_estimation
# 控制平滑程度，数值越大，越平滑
density_est.covariance_factor = lambda:  .3
density_est._compute_covariance()
x_data = np.arange(min(data), max(data), 200)


# 绘制密度估计曲线
def densityplot(x_data, density_est, x_label, y_label, title):

    _, ax = plt.subplots()
    ax.plot(x_data, density_est(x_data), color='#539caf', lw=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 调用绘图函数
densityplot(x_data=x_data,
            density_est=density_est,
            x_label='Check Outs',
            y_label='Frequency',
            title='Distribution of Registered Check Outs')
plt.show()
print(type(density_est))


# step 5 组间分析
# - 组间定量比较
# - 分组粒度
# - 组间据类
# 柱状图 一级类间均值方差比较
# 分天分析统计特征
mean_total_co_day = daily_data[['weekday', 'cnt']].groupby('weekday').agg([np.mean, np.std])
mean_total_co_day.columns = mean_total_co_day.columns.droplevel()


# 定义绘制柱状图函数
def barplot(x_data, y_data, error_data, x_label, y_label, title):

    _, ax = plt.subplots()
    # 柱状图
    ax.bar(x_data, y_data, color='#539caf', align='center')
    # 控制方差
    # ls='none'去掉bar之间的连线
    ax.errorbar(x_data, y_data, yerr=error_data, color='#297803', ls='none', lw=5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 绘图函数调用
barplot(x_data=mean_total_co_day.index.values,
        y_data=mean_total_co_day['mean'],
        error_data=mean_total_co_day['std'],
        x_label='Day of Week',
        y_label='Check Outs',
        title='Total Check Outs by Day of Week (0 = Sunday)')
plt.show()

print(mean_total_co_day)
print(mean_total_co_day.columns)


# 堆积柱状图 多级类间相对占比比较
# 分天统计注册和偶然使用的情况
mean_by_reg_co_day = daily_data[['weekday', 'registered', 'casual']].groupby('weekday').mean()
print(mean_by_reg_co_day)
# 分天统计注册和偶然使用的占比
mean_by_reg_co_day['total'] = mean_by_reg_co_day['registered'] + mean_by_reg_co_day['casual']
mean_by_reg_co_day['reg_prop'] = mean_by_reg_co_day['registered'] / mean_by_reg_co_day['total']
mean_by_reg_co_day['casual_prop'] = mean_by_reg_co_day['casual'] / mean_by_reg_co_day['total']


# 绘制堆积柱状图
def stackbarplot(x_data, y_data_list, y_data_names, colors, x_label, y_label, title):

    _, ax = plt.subplots()
    # 循环绘制堆积柱状图
    for i in range(len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color=colors[i], align='center', label=y_data_names[i])
        else:
            # 采用堆积方式，除了第一个分类，后面的分类都从前面一个分类的柱状图接着画
            # 用归一化保证最终累计结果为1
            ax.bar(x_data, y_data_list[i], color=colors[i], align='center',
                   label=y_data_names[i], bottom=y_data_list[i-1])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')  # 设定图例位置


# 调用绘图函数
stackbarplot(x_data=mean_by_reg_co_day.index.values,
             y_data_list=[mean_by_reg_co_day['reg_prop'], mean_by_reg_co_day['casual_prop']],
             y_data_names=['Registered', 'Casual'],
             colors=['#539caf', '#7663b0'],
             x_label='Day of Week',
             y_label='Propotion of Check Outs',
             title='Check Outs By Registration Status and Day of Week (0 = Sunday)')
plt.show()
# 工作日VS节假日，为什么会有这样的差别


# 分组柱状图 多级类间绝对值比较
# 绘制分组柱状图的函数
def groupedbarplot(x_data, y_data_list, y_data_names, colors, x_label, y_label, title):

    _, ax = plt.subplots()
    # 设置每一组柱状图的宽度
    total_width = 0.8
    # 设置每一个柱状图的宽度
    ind_width = total_width / len(y_data_list)
    # 计算每一个柱状图的中心偏移
    alteration = np.arange(-total_width/2+ind_width/2, total_width/2+ind_width/2, ind_width)
    # print(alteration)

    # 分别绘制每一个柱状图
    for i in range(0, len(y_data_list)):
        # 横向散开绘制
        ax.bar(x_data+alteration[i], y_data_list[i], color=colors[i], label=y_data_names[i], width=ind_width)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')


# 调用绘图函数
groupedbarplot(x_data=mean_by_reg_co_day.index.values,
               y_data_list=[mean_by_reg_co_day['registered'], mean_by_reg_co_day['casual']],
               y_data_names=['Registered', 'Casual'],
               colors=['#539caf', '#7663b0'],
               x_label='Day of Week',
               y_label='Check Outs',
               title='Check Outs by Registration Status and Day of Week (0 = Sunday)')
plt.show()


# 箱式图
# - 多级类间数据分析比较
# 柱状图+堆叠灰度图
# 只需要指定分类的标签，就能自动绘制箱式图
days = np.unique(daily_data['weekday'])
bp_data = []
for day in days:
    bp_data.append(daily_data[daily_data['weekday'] == day]['cnt'].values)
# print(daily_data['weekday'])
# print(days)
# print(bp_data)


# 定义绘图函数
def boxplot(x_data, y_data, base_color, median_color, x_label, y_label, title):

    _, ax = plt.subplots()
    # 设置样式
    ax.boxplot(y_data,
               # 箱子是否颜色填充
               patch_artist=True,
               # 中位数线颜色
               medianprops={'color': base_color},
               # 箱子颜色设置，color边框颜色，facecolor填充颜色
               boxprops={'color': base_color, 'facecolor': median_color},
               # 猫须颜色whisker
               whiskerprops={'color': median_color},
               # 猫须界限颜色whisker cap
               capprops={'color': base_color})

    # 箱图与x_data保持一致
    ax.set_xticklabels(x_data)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# 调用绘图函数
boxplot(x_data=days,
        y_data=bp_data,
        base_color='b',
        median_color='r',
        x_label='Day of Week',
        y_label='Check Outs',
        title='Total Check Outs by Day of Week (0 = Sunday)')
plt.show()


# 简单总结
# - 关联分析、数值比较：散点图、曲线图
# - 分布分析：灰度图、密度图
# - 涉及分类的分析：柱状图、箱式图
