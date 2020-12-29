# 推荐部分的逻辑

import numpy as np
import pandas as pd
import logging

import config

# 推荐部分的逻辑

class RecomApi():
    def __init__(self, threshold, lower=0.64, number=None):
        self.model_dict = {}
        self.threshold = threshold
        self.lower = lower

        if number:
            self.number = number
        else:
            try:
                from config import RECOM_NUMBER
                self.number = RECOM_NUMBER
            except:
                self.number = 10

    def _sort(self, index, distance, jacard, number=None, *args, **kwargs):
        '''
        返回排序后的索引和评分
        :param index: 索引
        :param distance: array对象, 欧氏距离
        :param jarcard: array对象，杰卡尔德系数
        :param number: 推荐人数
        :return:[{'index':index, 'similarity':simirity}, {...},...]
        '''
        if number:
            number = number
        else:
            number = self.number
        similarity = (20 - (distance - 5 * jacard)) / 25
        indexs = list(np.argsort(-similarity)[:number + 1])  # 降序排列

        # 去掉自身
        if index not in indexs:
            sort_index = indexs[:number]
        else:
            indexs.remove(index)
            sort_index = indexs
        # 去掉低于相似度过低的
        return [{"index": int(i), "similarity": float(similarity[i])} for i in sort_index if similarity[i] >= self.lower]

    def sort(self, gender, yxdm, index, distance, jacard, number=None, *args, **kwargs):
        data = self._sort(index=index, distance=distance,
                          jacard=jacard, number=number, *args, **kwargs)
        if not data:
            return {'code': 2, 'msg': '相似度过低'}
        for i in data:
            i.update({'gender': gender, 'yxdm': yxdm})
        return {'code': 0, 'list': data, 'num': len(data)}

    def get_model(self, key):
        if key not in self.model_dict:
            distance_model = DistanceModel(threshold=self.threshold)
            jacard_model = JacardModel(threshold=self.threshold)
            self.model_dict[key] = [distance_model, jacard_model]
        else:
            distance_model, jacard_model = self.model_dict.get(key)

        return distance_model, jacard_model

    def fit(self, index, yxdm, gender, q_1=None, q_2=None, q_3=None, q_4=None, q_5=None, label=None, number=None, *args,
            **kwargs):
        '''

        :param index: 索引
        :param yxdm: 学院代码
        :param gender: 性别
        :param q_1: 问卷1
        :param q_2: 问卷2
        :param q_3:
        :param q_4:
        :param q_5:
        :param label: 标签，set
        :param number: 推荐人数
        :param args:
        :param kwargs:
        :return: {'code':0, 'data':[{'index':index, 'similarity':simirity}, {...},...]}
        '''
        key = str(yxdm) + str(gender)
        distance_model, jacard_model = self.get_model(key)

        # 非第一次查询，不用进行预测，直接进行排序即可
        if index < distance_model.data_length:
            if not distance_model.is_matrix():
                return {'level': 'info', "code": 1, 'msg': '当前推荐人数较少'}
            return self.sort(gender, yxdm, index=index, distance=distance_model.get_matrix(index=index),
                             jacard=jacard_model.get_matrix(index=index), number=number, *args, **kwargs)

        result1 = distance_model.predict(index, [q_1, q_2, q_3, q_4, q_5])
        result2 = jacard_model.predict(index, [label, ])
        code1 = result1.get('code')
        code2 = result2.get('code')

        if code1 == code2 and code1 in config.code:
            if int(code1) != 0:
                return result1
            else:
                return self.sort(gender, yxdm, index=index, distance=distance_model.get_matrix(index=index),
                                 jacard=jacard_model.get_matrix(index=index), number=number, *args, **kwargs)
        else:
            return {"code": "-1", "msg": "模型状态码不一致", "level": "error"}

class Model():
    def __init__(self, threhold, columns, cross_value, *args, **kwargs):
        self.threhold = threhold
        self.cross_value = cross_value
        self.data = pd.DataFrame(columns=columns) if columns else pd.DataFrame()

    def is_matrix(self):
        '''
        是否生成了matrix
        :return: bool
        '''
        if hasattr(self, 'matrix'):
            return True

    def get_matrix(self, index):
        if hasattr(self, "matrix"):
            return self.matrix[index]
        else:
            raise AttributeError("ModelError：未生成matrix")

    @property
    def data_length(self):
        return self.data.shape[0]

    def predict(self, index, sample):
        if not hasattr(self, 'matrix'):
            self.append_data(index, sample)
            self.create_matrix()
            return {'level': 'info', "code": 1, 'msg': '当前推荐人数较少'}

        else:
            count = self.count(index, sample)

            shape = self.matrix.shape
            if shape[0] == shape[1]:  # 做一个检查，必须是方阵才能进行操作
                row = np.append(count, self.cross_value)
                row.shape = (1, row.shape[0])
                self.matrix = np.r_[np.c_[self.matrix, count], row]
                # 添加到data中
                self.append_data(index, sample)
                return {'code': 0, 'level': 'info', 'msg': '数据正常返回'}
            else:
                '''
                这里暂时没完成，标注，应该加一个应急方案
                '''
                # 如果不是方阵，给出最高级日志
                return {'code': 3, "level": "critical", "msg": "matrix矩阵不是方阵"}

    def append_data(self, index, sample):
        if index == self.data_length:
            self.data.loc[self.data_length] = sample
        elif index < self.data_length:
            return
        else:
            raise IndexError("index传参错误， index：{}".format(index))

    def create_matrix(self):
        if hasattr(self, 'matrix'):
            logging.error("matrix已存在")
            return

        if self.data_length >= self.threhold:
            logging.debug("创建matrix")
            self.matrix = np.ones((self.data_length, self.data_length)) * self.cross_value
            for index, sample in self.data.iterrows():
                count = self.count(index, sample)
                self.matrix[index, :] = count
                self.matrix[:, index] = count

            # 将自身调为100或0
            for i in range(self.data_length):
                self.matrix[i, i] = self.cross_value

    def count(self, index, sample):
        '''
        计算的逻辑
        :param index:
        :param sample:
        :return:
        '''
        pass

class DistanceModel(Model):
    '''
    距离的模型
    '''

    def __init__(self, threshold=10, cross_value=100):
        columns = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']
        super(DistanceModel, self).__init__(threshold, columns, cross_value=cross_value)

    def count(self, index, sample):
        data2 = self.data - sample
        data2 = data2.fillna(0)
        count = data2.apply(lambda x: x.dot(x), axis=1)
        return count

    # def predict(self, index, sample):
    #     # 判断是否生成了matrix
    #     if not hasattr(self, 'matrix'):
    #         # 将数据添加到data中
    #         self.append_data(index, sample)
    #
    #         # 填写问卷人数较少，否则直接生成matrix
    #         if self.data_length >= self.threhold:
    #             self.matrix = np.ones((self.data_length, self.data_length)) * 100
    #             for index, sample in self.data.iterrows():
    #                 data2 = self.data - sample
    #                 count = data2.apply(lambda x: x.dot(x), axis=1)
    #                 self.matrix[index, :] = count
    #                 self.matrix[:, index] = count
    #
    #             # 将自身调为1000（也就是将自身作为最后的）
    #             for i in range(self.data_length):
    #                 self.matrix[i, i] = 1000
    #
    #         return {'level': 'info', "code": 1, 'msg': '当前推荐人数较少'}
    #
    #     else:
    #         data2 = self.data - sample
    #         # 将数据添加到data中
    #         self.data.loc[self.data_length] = sample
    #         count = data2.apply(lambda x: x.dot(x), axis=1)
    #
    #         shape = self.matrix.shape
    #         if shape[0] == shape[1]:  # 做一个检查，必须是方阵才能进行操作
    #             row = np.append(count, 1000)
    #             row.shape = (1, row.shape[0])
    #             self.matrix = np.r_[np.c_[self.matrix, count], row]
    #
    #             return {'code': 0, 'level': 'info', 'msg': '数据正常返回'}
    #
    #         else:
    #             '''
    #             这里暂时没完成，标注，应该加一个应急方案
    #             '''
    #             # 如果不是方阵，给出最高级日志
    #             return {'code': 3, "level": "critical", "msg": "matrix矩阵不是方阵"}


class JacardModel(Model):
    def __init__(self, threshold=10, cross_value=0):
        columns = ['label']
        super(JacardModel, self).__init__(threshold, columns, cross_value=cross_value)

    def count(self, index, sample):
        count = self.data.iloc[:, 0].apply(lambda x: len((x & sample[0])) / (len((x | sample[0])) + 2 ))
        return count

    # def predict(self, index, sample):
    #     if not hasattr(self, 'matrix'):
    #         # 将数据添加到data中
    #         self.data.loc[self.data_length] = sample
    #
    #         # 判断是否生成了matrix
    #         if self.data_length >= self.threhold:
    #             self.matrix = np.ones((self.data_length, self.data_length)) * 100
    #             for index, sample in self.data.iterrows():
    #                 count = self.data.iloc[:, 0].apply(lambda x: len((x & sample[0])) / len((x | sample[0])))
    #                 self.matrix[index, :] = count
    #                 self.matrix[:, index] = count
    #
    #             # 将自身调为0（也就是将自身作为最后的）
    #             for i in range(self.data_length):
    #                 self.matrix[i, i] = 0
    #
    #         return {'level': 'info', "code": 1, 'msg': '当前推荐人数较少'}
    #
    #     else:
    #         count = self.data.iloc[:, 0].apply(lambda x: len((x & sample[0])) / len((x | sample[0])))
    #         shape = self.matrix.shape
    #         if shape[0] == shape[1]:  # 做一个检查，必须是方阵才能进行操作
    #             row = np.append(count, 0)
    #             row.shape = (1, row.shape[0])
    #             self.matrix = np.r_[np.c_[self.matrix, count], row]
    #
    #             # 将数据添加到data中
    #             self.data.loc[self.data_length] = sample
    #             return {'code': 0, 'data': self.matrix[-1]}
    #         else:
    #             # 如果不是方阵就只能重新计算
    #             return {'code': 3, "level": "critical", "msg": "matrix矩阵不是方阵"}