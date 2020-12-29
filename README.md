# 室友推荐-v1.0

## 推荐部分

主要逻辑在`roomate/recommend`文件中

## 后端部署

主要逻辑在`/app.py`和`handler/main.py`

定义了一个接口，用于进行推荐
```
('/recommend', main.RecommandHandler),
```


## 室友推荐
**请求方式：** POST

**url:** http:127.0.0.1:8013/recommend

**参数：** 

其中，index和YXDM以及XBDM是必填项
```json
{
  "index":"123", 
  "YXDM":"学院代码", 
  "XBDM":"性别", 
  "q_1":"q_1", 
  "q_2":"q_2", 
  "q_3":"q_3", 
  "q_4":"q_4", 
  "q_5":"q_5", 
  "label":"标签"}
```

**返回数据：**

成功数据返回
```json
{
    "list" : [{"index":"xxx", "yxdm":"xxx", "gender":"1", "similarity":"0.5"},{"index":"xxx", "yxdm":"xxx", "gender":"1", "similarity":"0.5"},{"index":"xxx", "yxdm":"xxx", "gender":"1", "similarity":"0.5"}],
    "code": 0,
    "num": 3
  }
```

其他情况(后续根据实际情况可能会有更改)
```json
{
  "level": "info", 
  "code":1, 
  "msg":"当前推荐人数较少"
  }
```

```json
{
  "code": 2, 
  "msg": "相似度过低"
  }
```