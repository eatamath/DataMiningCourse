import pandas as pd
import numpy as np
from numpy.linalg import *
import scipy as sp
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.neighbors import *
from collections import *
from pyclust import *
from sklearn.manifold import *
from sklearn.cluster import KMeans
from module.EKmeans import *
from module.CTree import *
import datetime
from collections import *

DATA_PATH = './trade_new.csv'

_col = {
    'uid':'uid',
    'tran_time':'sldatime', #
    'gender_age':'cmrid',
    'vipno':'vipno', #
    'itemno':'pluno', #
    'amount':'amt', #
    'quantity':'qty', #
    'brandno':'bndno', #
}

_col_class = {
    'class1':'class1',
    'class2':'class2',
    'class3':'class3',
    'class4':'class4',
    'class5':'class5',
}

_col2 ={**_col,**_col_class}

print('columns',_col2)

df = pd.read_csv(DATA_PATH)
df = df[_col.values()]
new_df = pd.DataFrame([],columns=_col2.values())





category = df[_col2['itemno']].values.astype('str')
category = np.array(list(map(lambda x:np.array([x[:2],x[:3],x[:4],x[:5],x[5:]]),category)))
category = pd.DataFrame(category,columns=_col_class.values())
if (category.index.start == df.index.start) and \
    (category.index.stop == df.index.stop):
    new_df = df.join(category)

df_amount_sum = new_df.groupby([_col2['vipno'],_col2['class1'],_col2['class2'],_col2['class3'],_col2['class4']],as_index=False).agg({
    _col2['amount']:sum
})
df_amount_sum3 = new_df.groupby([_col2['vipno'],_col2['class1'],_col2['class2'],_col2['class3']],as_index=False).agg({
    _col2['amount']:sum
})
df_amount_sum2 = new_df.groupby([_col2['vipno'],_col2['class1'],_col2['class2']],as_index=False).agg({
    _col2['amount']:sum
})
df_amount_sum1 = new_df.groupby([_col2['vipno'],_col2['class1']],as_index=False).agg({
    _col2['amount']:sum
})
df_amount = [df_amount_sum1, df_amount_sum2, df_amount_sum3, df_amount_sum]
del df
# df_amount_sum.to_csv('./result/a1-amount_sum.csv')




### 将时间转化为 time level 1-4
def processTimeLevel(new_df):
    global _col2
    _col2.update({
        'tran_time_level':'tran_time_level',
        'tran_date':'tran_date'
    })
    
    tran_time = new_df[ _col2['tran_time'] ]
    tran_time = list(map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timestamp(),tran_time))
    new_df[ _col2['tran_time'] ] = tran_time
    
    tran_date = list( map(lambda x:datetime.datetime.strftime(datetime.datetime.fromtimestamp(x),'%Y%m%d'),tran_time) ) ### to y-m-d
    new_df[ _col2['tran_date'] ] = tran_date
    
    df_temp = new_df[ [_col2['vipno'], _col2['tran_date']] ]
#     df_temp = df_temp.groupby(by=['vipno']).agg({_col2['tran_date']:max}).reset_index().values
#     df_temp = {x[0]:x[1] for x in df_temp}
    q = max(df_temp[ _col2['tran_date'] ])
    df_temp = {row['vipno']:q for idx,row in df_temp.iterrows()}
#     print(set(new_df['vipno'])-set(df_temp.keys()))
    tran_time_level = []
    for idx,row in new_df.iterrows():
        latest_date = q
        latest_date = datetime.datetime.strptime(latest_date,'%Y%m%d').timestamp()
        latest_date = datetime.datetime.fromtimestamp(latest_date)
        time_level4 = (latest_date - datetime.timedelta(days=30)).timestamp()
        time_level3 = (latest_date - datetime.timedelta(days=2*30)).timestamp()
        time_level2 = (latest_date - datetime.timedelta(days=4*30)).timestamp()
#         time_level1 = (latest_date - datetime.timedelta(days=16*30)).timestamp()
        time_levels = np.array([time_level2,time_level3,time_level4])
        row_time = row[_col2['tran_time']]
        tran_time_level.append( len(np.where(row_time>time_levels)[0])+1 ) ### append time level
        
    new_df[ _col2['tran_time_level'] ] = tran_time_level
    return new_df


new_df = processTimeLevel(new_df)
new_df = new_df.sort_values(by=[_col2['vipno']]) ### sort by customer

