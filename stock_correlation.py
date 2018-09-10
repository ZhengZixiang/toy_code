# -*- coding: utf-8 -*-
# 获取浦发银行和光大银行2016年股票数据，观察股票价格相关性

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts

# 浦发银行股票代码
s_pf = '600000'
# 光大银行股票代码
s_gd = '601818'
# 起始日期
start_date = '2016-01-01'
# 结束日期
end_date = '2016-12-31'
dframe_pf = ts.get_h_data(s_pf, start=start_date, end=end_date).sort_index(axis=0, ascending=True)
dframe_gd = ts.get_h_data(s_gd, start=start_date, end=end_date).sort_index(axis=0, ascending=True)

dframe = pd.concat([dframe_pf.close, dframe_gd.close], axis=1, keys=['pf_close', 'gd_close']) # close表示当日收盘价

# 针对休息日和停牌，则用前一天的股票收盘价填充
dframe.ffill(axis = 0, inplace = True)

# 增加归一化字段，让两条曲线的变化规律易于观察
dframe['pf_norm'] = dframe.pf_close / float(dframe.pf_close[0]) * 100
dframe['gd_norm'] = dframe.gd_close / float(dframe.gd_close[0]) * 100

dframe.to_csv('pf_gd.csv')

# 计算相关性
corr = dframe.corr(method = 'pearson', min_periods = 1)
print('\n', corr)

# 输出股票走势图
dframe.plot(figsize = (20, 12))
plt.savefig('stock_corr_norm.jpg')
plt.close()
