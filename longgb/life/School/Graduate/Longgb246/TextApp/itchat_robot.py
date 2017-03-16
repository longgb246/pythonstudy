#-*- coding:utf-8 -*-
import requests
import itchat
# from wxpy import *


KEY = '05b2a91b3e994819bb0a145001479934'

# 向api发送请求
def get_response(msg):
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key'    : KEY,
        'info'   : msg,
        'userid' : '18817360279',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        return r.get('text')
    except:
        return '听不懂你在说什么。'


# 注册方法
@itchat.msg_register(itchat.content.TEXT)
def tuling_reply(msg):
    # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
    defaultReply = u'我收到: ' + msg['Text']
    # 如果图灵Key出现问题，那么reply将会是None
    reply = get_response(msg['Text'])
    # a or b的意思是，如果a有内容，那么返回a，否则返回b
    return reply or defaultReply


# 为了让修改程序不用多次扫码,使用热启动
itchat.auto_login(hotReload=True)
itchat.run()

#
# #微信自动回复
# robot = Robot()
# # 回复来自其他好友、群聊和公众号的消息
# @robot.register()
# def reply_my_friend(msg):
#     message = '{}'.format(msg.text)
#     replys = talks_robot(info=message)
#     return replys
#
# # 开始监听和自动处理消息
# robot.start()
