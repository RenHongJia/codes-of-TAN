from __future__ import division

__author__ = 'Victor Ruiz, vmr11@pitt.edu'

import pandas as pd
import numpy as np
from math import log
import random
from sklearn.metrics.cluster import supervised as sd

def entropy(data_classes, base=2):
# def entropy(data_classes,feature_number=None, base=2):
    '''
    计算熵H(X)
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    :return: value of entropy
    '''
    # if not isinstance(data_classes, pd.core.series.Series):
    #     raise AttributeError('input array should be a pandas series')
    # unique()将类别转化为numpy.ndarray形式[1,2,3]
    classes = np.unique(data_classes)
    # 样本个数N_data
    N_data = len(data_classes)
    N_class = len(classes)
    ent = 0  # initialize entropy
    Lap_smo_factor = 1

    # iterate over classes 遍历每个类别
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        # 拉普拉斯平滑 Laplace smoothing
        proportion = (len(partition)+Lap_smo_factor) / (N_data + N_class)
        #update entropy
        ent -= proportion * log(proportion, base)
    """
    if N_class==feature_number:
        return ent
    else:
        left_n = range(feature_number-N_class)
        for i in left_n:
            p = 1/feature_number
            ent -= p * log(p, base)
    """

    return ent
def con_entropy(X,Y,base=2):
    '''
    计算条件熵H(X/Y)
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    if not isinstance(X&Y, pd.core.series.Series):
        raise AttributeError('input array should be a pandas series')
    # unique()将类别转化为numpy.ndarray形式[1,2,3]
    classes = X.unique()
    # 样本个数N
    N_data = len(data_classes)
    ent = 0  # initialize entropy
    # N_data = len(data_classes)
    # N_class = len(classes)
    # ent = 0  # initialize entropy
    # Lap_smo_factor = 1
    # iterate over classes 遍历每个类别
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def mutual_infor_score(label_x, label_y):
    '''
    计算互信息I(X;Y)
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    label_x : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    label_x : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    '''
    labels_true, labels_pred = sd.check_clusterings(label_x, label_y)
    # Merge labels_true and labels_pred
    data_union  = pd.DataFrame({"x_col":labels_true,"y_col":labels_pred})
    # compute H(x)
    feature_number= len(np.unique(labels_true))
    ENT = entropy(labels_true)
    # ENT = entropy(labels_true,feature_number=feature_number)
    #
    """
    样本个数 data_N;
    x_N：x取值种类数
    y_N：y取值种类数
    class_x :x类别  numpy.ndarray  ['a1' 'a2']
    class_y :y类别
    ENT_X_Y :条件熵H(X|Y) 
    Lap_smo_factor:拉普拉斯平滑
    """
    data_N = len(labels_true)
    class_x=np.unique(labels_true)
    class_y = np.unique(labels_pred)
    x_N = len(class_x)
    y_N = len(class_y)
    ENT_X_Y = 0
    Lap_smo_factor = 1
    for y in class_y:

        dataXy = data_union[data_union["y_col"]==y]
        len_y = len(dataXy)
        data_x_col = np.asanyarray(dataXy["x_col"])
        entXy = entropy(data_x_col)
        # entXy = entropy(data_x_col,feature_number=x_N)
        p_y = (len_y+Lap_smo_factor)/(y_N+data_N)
        ENT_X_Y = ENT_X_Y+entXy*p_y

    mi = ENT  -  ENT_X_Y
    return mi
def mutual_information(label_x, label_y):
    '''
    计算互信息I(X;Y)
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    label_x : int array, shape = [n_samples]
        A clustering of the data into disjoint subsets.

    label_x : array, shape = [n_samples]
        A clustering of the data into disjoint subsets.
    '''
    # if not isinstance(dataset, pd.core.frame.DataFrame):
    #     raise AttributeError('input dataset should be a pandas data frame')
    labels_true, labels_pred = sd.check_clusterings(label_x, label_y)
    # 合并 labels_true, labels_pred
    data_union  = pd.DataFrame({"x_col":labels_true,"y_col":labels_pred})
    # 计算H(x)
    feature_number= len(np.unique(labels_true))
    """
      样本个数 data_N;
      x_N：x取值种类数
      y_N：y取值种类数
      class_x :x类别  numpy.ndarray  ['a1' 'a2']
      class_y :y类别
      ENT_X_Y :条件熵H(X|Y) 
      Lap_smo_factor:拉普拉斯平滑
    """
    data_N = len(labels_true)
    class_x=np.unique(labels_true)
    class_y = np.unique(labels_pred)
    x_N = len(class_x)
    y_N = len(class_y)
    xy_N = x_N*y_N
    ENT_X_Y = 0
    Lap_smo_factor = 1
    # ENT = entropy(labels_true,feature_number=feature_number)
    base =2
    for x in class_x:
        for y in class_y:
            data_xy = data_union[(data_union["y_col"]== y)&(data_union["x_col"] ==x)]
            xy_number = len(data_xy)

            p_xy = (xy_number+Lap_smo_factor)/(data_N+xy_N)
            data_y = data_union[data_union["y_col"] == y]
            data_x = data_union[data_union["x_col"] == x]
            y_number = len(data_y)
            x_number = len(data_x)
            p_y = (y_number+Lap_smo_factor)/(data_N+y_N)
            p_x = (x_number+Lap_smo_factor)/(data_N+x_N)
            kl=p_xy*log(p_xy/(p_y*p_x), base)
            ENT_X_Y = ENT_X_Y+kl
    return ENT_X_Y
def con_mutual_infor_score(x, y, z):
    '''
    计算条件互信息I(X;Y|Z)
    '''
    label_x, label_y = sd.check_clusterings(x, y)
    label_z = z
    # 合并 labels_true, labels_pred
    data_union = pd.DataFrame({"x_col": label_x, "y_col": label_y,"z_col": label_z})
    """
      样本个数 data_N;
      x_N：x取值种类数
      y_N：y取值种类数
      z_N：y取值种类数
      class_x :x类别  numpy.ndarray  ['a1' 'a2']
      class_y :y类别
      class_z :z类别
      Lap_smo_factor:拉普拉斯平滑
    """
    data_N = len(label_x)
    class_x = np.unique(label_x)
    class_y = np.unique(label_y)
    class_z = np.unique(label_z)
    x_N = len(class_x)
    y_N = len(class_y)
    z_N = len(class_z)
    xyz_N = x_N * y_N * z_N
    xz_N = x_N *  z_N
    yz_N = x_N  * z_N
    con_mu_infor_score = 0
    Lap_smo_factor = 1
    # ENT = entropy(labels_true,feature_number=feature_number)
    base = 2
    for z in class_z:
        for x in class_x:
            for y in class_y:
                data_xyz = data_union[
                    (data_union["y_col"] == y) & (data_union["x_col"] == x) & (data_union["z_col"] == z)]
                data_xz = data_union[(data_union["x_col"] == x) & (data_union["z_col"] == z)]
                data_yz = data_union[(data_union["y_col"] == y) & (data_union["z_col"] == z)]
                data_z = data_union[(data_union["z_col"] == z)]
                xyz_number = len(data_xyz)
                xz_number = len(data_xz)
                yz_number = len(data_yz)
                z_number = len(data_z)

                p_xyz = (xyz_number + Lap_smo_factor) / (data_N + xyz_N)
                p_xz = (xz_number + Lap_smo_factor) / (data_N + xz_N)
                p_yz = (yz_number + Lap_smo_factor) / (data_N + yz_N)
                p_z = (z_number + Lap_smo_factor) / (data_N + z_N)
                con_mu_infor_score = con_mu_infor_score + p_xyz * log((p_xyz * p_z) / (p_xz * p_yz), base)
    return con_mu_infor_score

def get_predict_score(predict_data, label):
    """

    :param predict_data : 用模型预测的变量值
    :param label: 真实值
    :return: score 预测得分
    """
    score = get_predict(predict_data, label)
    result = 0
    for i in score :
        if i=="total":
            result =score["total"]

    return result
def get_predict(predict_data, label):
    """
    :param self:
    :param predict_data: series 类型
    :param label:series 类型
    :return: score 字典形式
    """

    predict_data = predict_data[predict_data.columns[0]]
    x_list = list(predict_data)
    y_list = list(label)
    number = list(sorted(np.unique(label)))
    # 合并 labels_true, labels_pred
    data_union = pd.DataFrame({"x_col": x_list, "y_col": y_list})
    score = {}
    right = 0
    total = 0
    for i in number:
        data_xy = data_union[(data_union["y_col"] == i) & (data_union["x_col"] == i)]
        data_y = data_union[(data_union["y_col"] == i)]
        # xy_number:x=y=i的个数
        # y_number: y=i的个数
        xy_number = len(data_xy)
        y_number = len(data_y)
        right = right+ xy_number
        total = total +y_number
        wrong = y_number - xy_number
        score[i] = [xy_number/y_number,xy_number,y_number]
    score["total"]=[right/total,right,total]
    return score
