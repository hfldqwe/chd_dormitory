import json

import tornado.web
import logging

from config import THRESHOLD
from roomate.recommend import RecomApi


recom = RecomApi(threshold=THRESHOLD)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("这是一个测试页面：室友推荐")

class RecommandHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.write("你想要一个什么样的室友呢。。")

    def post(self, *args, **kwargs):
        global recom
        '''
        {'index':1, 'yxdm':'123456', 'gender':1, 'q_1':1, 'q_2':2, 'q_3':2, 'q_4':1, 'q_5':1, 'label':'a,b,c,d'}
        :param args:
        :param kwargs:
        :return: {
            "list" : [{'index':index, 'yxdm':yxdm, 'gender':gender, 'similarity':0.5},{'index':index, 'yxdm':yxdm, 'gender':gender, 'similarity':0.5},{'index':index, 'yxdm':yxdm, 'gender':gender, 'similarity':0.5}],
            'code': 0,
            'num': 3
        }
        '''
        # 解析json数据
        try:
            sample = json.loads(self.request.body)
        except:
            error_info = {"code":0, "msg":"数据解析失败"}
            self.write(error_info)
            self.finish()
            return
        clear, index, yxdm, gender, q_1, q_2, q_3, q_4, q_5, label = self.prepare_data(sample=sample)

        if clear:
            recom = RecomApi(threshold=THRESHOLD)
            self.finish("data已重置")
            return
        result = recom.fit(index, yxdm, gender, q_1=q_1, q_2=q_2, q_3=q_3, q_4=q_4, q_5=q_5, label=label, number=None)

        if result['code'] == 0:
            for data in result['list']:
                data['index'] = data['index'] + 1
        result = json.dumps(result, ensure_ascii=False)
        self.write(result)

    def prepare_data(self, sample):
        clear = sample.get("clear", None)
        index = sample.get("index", None)
        yxdm = sample.get("YXDM", None)
        gender = sample.get("XBDM", None)
        if clear:  # 重复查询不需要处理选项
            logging.debug("重复查询")
            return clear, index, yxdm, gender, None, None, None, None, None, None

        q_1 = int(sample.get("q_1", -1))
        q_2 = int(sample.get("q_2", -1))
        q_3 = int(sample.get("q_3", -1))
        q_4 = int(sample.get("q_4", -1))
        q_5 = int(sample.get("q_5", -1))
        label = sample.get("label", "")

        q_1 = int(q_1)-1 if q_1 and q_1 in [0,1,2] else None
        q_2 = int(q_2)-1 if q_2 and q_2 in [0,1,2,"0","1","2"] else None
        if q_3 == 1:
            q_3 = 4
        q_3 = (int(q_3)-2)/2 if q_3 in [0,2,4] else None
        q_4 = int(q_4)-1 if q_4 in [0,1,2] else None
        q_5 = (int(q_5)-0.5)*2 if q_5 in [0,1] else None

        index = int(index) - 1
        if label:
            label = label.split(",")
            label = set(i for i in label if i)
        else:
            label = set()

        return clear, index, yxdm, gender, q_1, q_2, q_3, q_4, q_5, label