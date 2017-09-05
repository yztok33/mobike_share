# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:44:05 2017
加入方位角degree---引自offline_lgb.py
@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:19:11 2017
LIGHTGBM 线上线下版
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:23:53 2017
LightGBM 版本
@author: Administrator
"""
'''
加入时间特征，

'''
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:52:25 2017
副本，若成功运行，则复制到正本中
问题：线上test到底需不需要get_label（）??
改了 get_distance ，用精确的地址---->成绩上升
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:45:15 2017
只做线下测试
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:16:40 2017
填补空缺位置,将min0826代码排版
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:02:40 2017
正式加入时间特征
@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:26:51 2017
意识到make_train_set中可能真的不能有重复，所以改掉训练样本
#结果确实得到提升！！不能重复
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:50:42 2017
加入 线上线下函数
@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:44:05 2017
加入方位角degree---引自offline_lgb.py
因为效果好，所以重新拉回offline_lgb.py
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:23:53 2017
LightGBM 版本
@author: Administrator
"""
'''
加入时间特征，

'''
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:52:25 2017
副本，若成功运行，则复制到正本中
问题：线上test到底需不需要get_label（）??
改了 get_distance ，用精确的地址---->成绩上升
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:45:15 2017
只做线下测试
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:16:40 2017
填补空缺位置,将min0826代码排版
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:02:40 2017
正式加入时间特征
@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:26:51 2017
意识到make_train_set中可能真的不能有重复，所以改掉训练样本
#结果确实得到提升！！不能重复
@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:50:42 2017
加入 线上线下函数
@author: Administrator
"""
import winsound
import os
import gc
import time
from datetime import datetime
import pickle
import geohash
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from math import radians, cos, sin, atan2, degrees
#单独给她一个文件夹
path='C:/Users/Administrator/Desktop/competition/mobike/online_offline'
os.chdir(path)

cache_path = 'mobike_cache_'
train_path = 'C:/Users/Administrator/Desktop/competition/mobike/train.csv'
test_path = 'C:/Users/Administrator/Desktop/competition/mobike/test.csv'
flag = True


# 补位方案1： 速度巨慢
def fill_hotloc(train,result):
    hotloc_count=train.groupby(['geohashed_end_loc'],as_index=False)['userid'].agg({'hotloc_count':'count'})
    hotloc=list(hotloc_count.sort_values(['hotloc_count']).tail(3).geohashed_end_loc)
    
    result['null_cnt'] = pd.isnull(result).sum(axis=1)
    for i in range(0,len(result)):
        if result.null_cnt[i]!=0:
            to_fill = hotloc[0:result.null_cnt[i]]
            if len(to_fill) !=  1:
                result.iloc[i,-result.null_cnt[i]-2:3] = to_fill
            else:
                result.iloc[i,-result.null_cnt[i]-2:3] = to_fill[0]    
    result.drop(['null_cnt'],axis=1,inplace=True)  
    return result




# 测评函数
def map(result):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test['geohashed_end_loc'] = np.nan
        merge = pd.concat([train,test])
        true = dict(zip(merge['orderid'].values,merge['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+')) #该函数将字典中的数据存到文件中
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    return score

# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

def get_degree(latA, lonA, latB, lonB):  
    radLatA = radians(latA)  
    radLonA = radians(lonA)  
    radLatB = radians(latB)  
    radLonB = radians(lonB)  
    dLon = radLonB - radLonA  
    y = sin(dLon) * cos(radLatB)  
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)  
    brng = degrees(atan2(y, x))  
    brng = (brng + 360) % 360  
    return brng  

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    #tao---得到每一个订单目的地概率排序值（0，1，2...）
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 对结果进行整理
def reshape(pred):
    result = pred[['orderid','geohashed_end_loc','pred']].copy() #我小小改动了
    result = rank(result,'orderid','pred',ascending=False)
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    #tao---针对三个变量时特别有用
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result



# 获取真实标签   #tao---这个对我们来说不重要--唉，是很有用啊,以后放着，不用删，可以复制
def get_label(data):     
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)   
        test['geohashed_end_loc'] = np.nan
        merge = pd.concat([train,test])
        #tao---不用.values可得到同样结果
        true = dict(zip(merge['orderid'].values, merge['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data['label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    return data


####################构造负样本##################

# 将用户骑行过目的的地点加入成样本   gaoding--根据历史，订单可能会到的终点
def get_user_end_loc(train,test):
    result_path = cache_path + 'user_end_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid','geohashed_end_loc']].drop_duplicates()
        result = pd.merge(test[['orderid','userid']],user_eloc,on='userid',how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 将用户骑行过出发的地点加入成样本    gaoding
def get_user_start_loc(train,test):
    result_path = cache_path + 'user_start_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_sloc = train[['userid', 'geohashed_start_loc']].drop_duplicates()
        result = pd.merge(test[['orderid', 'userid']], user_sloc, on='userid', how='left')
        #tao--因为起点也是可能成为终点的，因此再把起点中涉及到的地点也加进去作为可能的终点
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点去向最多的3个地点   gaoding--4个呢？5个呢？全加进去啊（效果可能不好）！到时候去重即可
def get_loc_to_loc(train,test):
    result_path = cache_path + 'loc_to_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
        sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(3)
        result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='left')
        #tao--test中订单号所对应的三个最常去的终点--这个feature重点看起点与终点的关系，与用户无关
        result = result[['orderid', 'geohashed_end_loc']]   
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



##############  构造特征 #############
# 获取用户历史行为次数  tao--去过多少次
def get_user_count(train,result):
    #tao--切完片，数据没少过，相当于安某一标准排一下数据，而count在这里只做计数，和你提那个变量无关
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
    result = pd.merge(result,user_count,on=['userid'],how='left')
    return result

# 获取用户去过某个地点历史行为次数
def get_user_eloc_count(train, result):
    user_eloc_count = train.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_count':'count'})
    result = pd.merge(result,user_eloc_count,on=['userid','geohashed_end_loc'],how='left')
    return result

# 获取用户从某个地点出发的行为次数
def get_user_sloc_count(train,result):
    user_sloc_count = train.groupby(['userid','geohashed_start_loc'],as_index=False)['userid'].agg({'user_sloc_count':'count'})
    #tao---用rename的原因是终点的种类多，防止用起点时出现对应不上的情形，从而出现NAN？
    #到时候改回去，然后重新建文件夹跑一边，看成绩如何 --成绩不行，改回来看看--成绩竟然一模一样，那就改成我的样子--
    #噢，理解了，因为万一这是result中用户的新起点，配对不上会缺省，造成浪费，所以改名
    user_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
    result = pd.merge(result, user_sloc_count, on=['userid', 'geohashed_end_loc'], how='left')
    return result

# 获取用户从这个路径走过几次  tao----特征：起点-终点走过几次：sloc_eloc_count
def get_user_sloc_eloc_count(train,result):
    user_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_sloc_eloc_count':'count'})
    result = pd.merge(result,user_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 获取用户从这个路径折返过几次   #tao--我觉得有问题，应该是反了,第二句不是多余，很有用，表明折回过几次
def get_user_eloc_sloc_count(train,result):
    user_eloc_sloc_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    #同上而已，user_eloc_sloc_count = train.groupby(['userid','geohashed_end_loc','geohashed_start_loc'],as_index=False)['userid'].agg({'user_eloc_sloc_count':'count'})
    user_eloc_sloc_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
    result = pd.merge(result,user_eloc_sloc_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result

# 计算两点之间的欧氏距离
def get_distance_degree(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(geohash.decode_exactly(loc)[:2])
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    degree = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        lat1,lon1,lat2,lon2=float(lat1),float(lon1),float(lat2),float(lon2)
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
        degree.append(get_degree(lat1,lon1,lat2,lon2))
    result.loc[:,'distance'] = distance
    result.loc[:,'degree'] = degree
    return result

# 获取目标地点的热度(目的地)
def get_eloc_count(train,result):
    eloc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_count': 'count'})
    result = pd.merge(result, eloc_count, on='geohashed_end_loc', how='left')
    return result

# 获取目标地点的热度(出发地)
def get_eloc_as_sloc_count(train,result):
    eloc_as_sloc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'eloc_as_sloc_count': 'count'})
    #tao--你看，下面这句话不就是bug吗，没用，所以前面应该是出现bug了
    #na---第三次，修改如下：start--end---成绩没我的好---再次修改回去
#    eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'})
    result = pd.merge(result, eloc_as_sloc_count, on='geohashed_start_loc', how='left')
    return result

# tao--提取时间信息
def time_info(one_time):
    d,h,m,s=time.strptime(one_time,"%Y-%m-%d %H:%M:%S")[2:6]
    week=datetime.strptime(one_time,"%Y-%m-%d %H:%M:%S").weekday()+1
    return d,h,m,s,week

# 构造时间特征      读取完数据的时候立即执行:分两次做。train，test
def get_time_features(data):
    #出于对test集时间不规范作出的处理
    data['starttime']=data['starttime'].str.split('.').str[0]
    a=DataFrame([(time_info(x)) for x in data.starttime],columns=['day','hour','minute','second','week'],index=data.index)
    data=pd.concat([data,a],axis=1)
    data['minute_c']=[(0<=x<30 and 1) or 0 for x in data['minute']]
    data['hour_sec']=[str(x)+'_'+str(y) for x,y in zip(data['hour'],data['minute_c'])]
    data['mor_aft_eve']=[(5<=x<11 and 1) or (11<=x<17 and 2) or 0 for x in data['hour']]
    data['workday']=[(1<=x<6 and 1) or 0 for x in data['week']] 
    #格式转换
    le=LabelEncoder()
    data['hour_sec']=le.fit_transform(data['hour_sec'])
    return data

# 获取用户起点小时段统计
def get_user_minute_c_count(train,result):
    user_minute_c_count=train.groupby(['userid','minute_c'],as_index=False)['userid'].agg({'user_minute_c_count':'count'})
    result=pd.merge(result,user_minute_c_count,on=['userid','minute_c'],how='left')
    return result

# 获取用户起点时段统计
def get_user_hour_sec_count(train,result):
    user_hour_sec_count=train.groupby(['userid','hour_sec'],as_index=False)['userid'].agg({'user_hour_sec_count':'count'})
    result=pd.merge(result,user_hour_sec_count,on=['userid','hour_sec'],how='left')
    return result

# 获取用户起点早中晚时段统计
def get_user_mor_aft_eve_count(train,result):
    user_mor_aft_eve_count=train.groupby(['userid','mor_aft_eve'],as_index=False)['userid'].agg({'user_mor_aft_eve_count':'count'})
    result=pd.merge(result,user_mor_aft_eve_count,on=['userid','mor_aft_eve'],how='left')
    return result

# 获取用户起点是否工作日统计
def get_user_workday_count(train,result):
    user_workday_count=train.groupby(['userid','workday'],as_index=False)['userid'].agg({'user_workday_count':'count'})
    result=pd.merge(result,user_workday_count,on=['userid','workday'],how='left')
    return result

# 获取用户出发时段统计
def get_user_eloc_workday_count(train,result):
    user_eloc_workday_count=train.groupby(['userid','geohashed_end_loc','workday'],as_index=False)['workday'].agg({'user_eloc_workday_count':'count'})
    result=pd.merge(result,user_eloc_workday_count,on=['userid','geohashed_end_loc','workday'],how='left')
    return result   

def get_eloc_workday_count(train,result):
    eloc_workday_count=train.groupby(['geohashed_end_loc','workday'],as_index=False)['workday'].agg({'eloc_workday_count':'count'})
    result=pd.merge(result,eloc_workday_count,on=['geohashed_end_loc','workday'],how='left')
    return result






# 获取起点-终点的热度
#########################################################
# 构造样本
def get_sample(train,test):
    result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(train, test)      # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_end_loc']
        user_start_loc = get_user_start_loc(train, test)  # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_start_loc']
        loc_to_loc = get_loc_to_loc(train, test)          # 筛选起始地点去向（终点）最多的3个地点
        # 汇总样本id   tao--也就是将上面三个情况中的订单-终点情况全部纳入，不遗漏
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],  
                            ]).drop_duplicates()
        # 根据end_loc添加标签(0,1)--把多分类问题转成了分类（概率）类型
        test_temp = test.copy()
        test_temp.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
        result = pd.merge(result, test_temp, on='orderid', how='left')
        result['label'] = (result['label'] == result['geohashed_end_loc']).astype(int)
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        #tao--这个~是取非的意思，其实可以用notnull,即test中全新的起点是得不到endloc,去掉对应的订单
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        #tao--此时这个result变量比test多了加长版的geohashed_end_loc，以及改名的label
    return result

# 制作训练集
def make_train_set(train,test):
    result_path = 'data_set_fixed_feature_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
        print('已添加时间特征...')
        print('已构造样本...')
        print('已构造特征...')
        print('result.columns:\n{}'.format(result.columns))
        print('已添加真实label')
        print('构造完毕')
    else:   
        print('开始添加时间特征...')
        train = get_time_features(train)
        test = get_time_features(test)
        print('开始构造样本...')
        result = get_sample(train,test)                                         # 构造备选样本
    
        print('开始构造特征...')
        result = get_user_count(train,result)                                   # 获取用户历史行为次数
        result = get_user_eloc_count(train, result)                             # 获取用户去过这个地点几次
        result = get_user_sloc_count(train, result)                             # 获取用户从目的地点出发过几次
        result = get_user_sloc_eloc_count(train, result)                        # 获取用户从这个路径走过几次
        result = get_user_eloc_sloc_count(train, result)                        # 获取用户从这个路径折返过几次
        #噢，明白了，作者直接把所有可能的终点都放进去了，预测反正逃不出这些点，所以算距离了
        result = get_distance_degree(result)                                           # 获取起始点和最终地点的欧式距离
        result = get_eloc_count(train, result)                                  # 获取目的地点的热度(目的地)
        result = get_eloc_as_sloc_count(train, result)                          # 获取目的地点的热度(出发地)
        #时间统计特征
        result = get_user_minute_c_count(train,result)
        result = get_user_hour_sec_count(train,result)
        result = get_user_mor_aft_eve_count(train,result)
        result = get_user_workday_count(train,result)
        result = get_user_eloc_workday_count(train,result)
        result = get_eloc_workday_count(train,result)
        result.fillna(0,inplace=True)
        print('result.columns:\n{}'.format(result.columns))
        print('添加真实label...')
        result = get_label(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        print('构造完毕')

    return result
 






####  温馨提示：线上线下的 训练集最好统一，这样才知道模型的好坏  #####
### 只有把训练集弄好，弄稳健，抗扰动之后，对test集的预测才能更准

# 18 features
import lightgbm as lgb

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 63,
    'learning_rate': 0.1,  #之前0.08
    'feature_fraction': 0.8, #我之前0.9
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50, # default=20
    'min_sum_hessian_in_leaf': 5,  #之前没有
    'verbose': 0 ,
    'seed': 201708,
    'scale_pos_weight': 10
}



# 训练提交
is_online=1

if is_online:
    t0 = time.time()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    #train=get_time_features(train)
    #test=get_time_features(test)
    train1 = train[(train['starttime'] < '2017-05-21 00:00:00')]
    train2 = train[(train['starttime'] >= '2017-05-21 00:00:00')]
    #train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
    #train2 = train[(train['starttime']>= '2017-05-23 00:00:00')]
    #train2.loc[:,'geohashed_end_loc'] = np.nan
    test.loc[:,'geohashed_end_loc'] = np.nan
    
    print('构造训练集')
    #到时候train_feat用后半段（多一点，可能会更好）
    train_feat = make_train_set(train1,train2)
    #过滤10公里以上的数据   
    train_feat=train_feat[train_feat.distance<=10000]

    print('构造线上测试集')
    test_feat = make_train_set(train,test)
    del train,test,train1,train2
    
    predictors=['user_count', 'user_eloc_count', 'user_sloc_count',
       'user_sloc_eloc_count', 'distance', 'degree',
       'eloc_count', 'eloc_as_sloc_count', 'user_minute_c_count',
       'user_hour_sec_count', 'user_mor_aft_eve_count', 'user_workday_count',
       'user_eloc_workday_count', 'eloc_workday_count']
    
    train_X=train_feat[predictors]
    train_y=train_feat['label']
    test_X=test_feat[predictors]
    test_y=test_feat['label']
    
    lgbtrain = lgb.Dataset(train_X, train_y)
    lgbtest = lgb.Dataset(test_X,test_y,reference=lgbtrain)
    del train_feat,train_X,train_y
    
    gbm = lgb.train(params, lgbtrain, num_boost_round=200,verbose_eval=10, 
                    valid_sets=[lgbtrain],early_stopping_rounds=10)
                    #feature_name=predictors) dataframe形式可省
    
    del lgbtrain
    gc.collect()
    
    test_feat.loc[:,'pred'] = gbm.predict(test_X)#num_iteration=gbm.best_iteration 若有，则模型自动选迭代次数最少的了
    result = reshape(test_feat)
    test = pd.read_csv(test_path)
    result = pd.merge(test[['orderid']],result,on='orderid',how='left')
    #填补
    result.fillna('0',inplace=True)
    
    result.to_csv('result.csv',index=False,header=False)
    import zipfile
    with zipfile.ZipFile("result.zip", "w") as fout:
        fout.write("result.csv", compress_type=zipfile.ZIP_DEFLATED)
#    fet_value=dict(zip(gbm.feature_name(),gbm.feature_importance()))
#    print(sorted(fet_value.items(), key=lambda e:e[1], reverse=True))
    print('一共用时{}秒'.format(time.time()-t0))
    winsound.PlaySound('SystemExclamation',winsound.SND_ALIAS)
    
else:
    t0 = time.time()
    train = pd.read_csv(train_path)
      #加入时间信息的train
#    test = pd.read_csv(test_path)
#电脑内存不够大，没办法，train1
    train1 = train[(train['starttime'] < '2017-05-19 00:00:00')]
    train2 = train[(train['starttime'] >= '2017-05-19 00:00:00')&(train['starttime'] < '2017-05-23 00:00:00')]
    train1_2 = train[(train['starttime'] < '2017-05-23 00:00:00')]
    train3 = train[(train['starttime'] >= '2017-05-23 00:00:00')] #就当成test，即假定原train没有这部分数据
    del train
    #train2 = train.copy()
    train3.loc[:,'geohashed_end_loc'] = np.nan
    #test.loc[:,'geohashed_end_loc'] = np.nan
    
    print('构造训练集')
    train_feat = make_train_set(train1,train2)
    #保存一次，下次单独拿来研究数据集用
    #train_feat.to_hdf('train_feat_0828.hdf', 'w', complib='blosc', complevel=5)

#过滤10公里以上的数据   
#    train_feat=train_feat[train_feat.distance<=10000]

    print('构造线下测试集')
    test_feat = make_train_set(train1_2,train3)

    del train1_2,train1,train2  #train3--相当于test,还不能删
    
# 23 features  remove some   相比线上多个  hour
    predictors =  [ 'hour',  'degree',
       'user_count', 'user_eloc_count', 'user_sloc_count',
       'user_sloc_eloc_count',  'distance',
       'eloc_count', 'eloc_as_sloc_count', 'user_minute_c_count',
       'user_hour_sec_count', 'user_mor_aft_eve_count', 'user_workday_count',
       'user_eloc_workday_count', 'eloc_workday_count']
    
    train_X=train_feat[predictors]
    train_y=train_feat['label']
    test_X=test_feat[predictors]
    test_y=test_feat['label']
    
    lgbtrain = lgb.Dataset(train_X, train_y)
    lgbtest = lgb.Dataset(test_X,test_y,reference=lgbtrain)
    del train_feat,train_X,train_y
    
    gbm = lgb.train(params, lgbtrain, num_boost_round=120,verbose_eval=10, 
                    valid_sets=[lgbtest],early_stopping_rounds=10)
                    #feature_name=predictors) dataframe形式可省
    
    del lgbtrain
    gc.collect()
    
    test_feat.loc[:,'pred'] = gbm.predict(test_X)#num_iteration=gbm.best_iteration 若有，则模型自动选迭代次数最少的了
    result = reshape(test_feat)
    
    #test = pd.read_csv(test_path)
    result = pd.merge(train3[['orderid']],result,on='orderid',how='left')
    result.fillna('0',inplace=True)
    result.to_csv('result_offline.csv',index=False,header=False)
    fet_value=dict(zip(gbm.feature_name(),gbm.feature_importance()))
    print(sorted(fet_value.items(), key=lambda e:e[1], reverse=True))
    print('score: {}'.format(map(result))) 
    print('一共用时{}秒'.format(time.time()-t0))
    winsound.PlaySound('SystemExclamation',winsound.SND_ALIAS)
    
'''
线上成绩：
[200]   training's auc: 0.931048
一共用时3656.815233707428秒
score: 0.27759732587251

过滤了距离的模型
[200]   training's auc: 0.916354
一共用时1104.5511767864227秒
score：

正确数据大小的模型：
[200]   training's auc: 0.919676  怎么与下面一样？
一共用时1102.890419960022秒
score: 

过滤了距离的 正确数据大小的 模型
[200]   training's auc: 0.919676
一共用时3443.2529010772705秒
score:
   
'''