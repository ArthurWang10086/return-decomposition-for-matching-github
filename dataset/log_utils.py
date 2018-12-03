#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "xuesong"
__copyright__ = "Copyright 2018, The NSH Anti-Plugin Project"
__version__ = "1.0.0"
__email__ = "xuesong@corp.netease.com"
__phone__ = "xuesong"
__description__ = "log tool"
__usage1__ = "see the description of class"

import os
import logging
import logging.handlers

# 日志级别
LOG_LEVEL = logging.DEBUG
# 日志格式
LOG_FORMAT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s - %(message)s'


class LogUtils(object):
    """log 工具类
    log file path = logs/*.log
    log level = debug

    Use:
        from CommonUtils.LogUtils import LogUtils
        logger = LogUtils(__file__).get_logger()
        logger.debug('test')
    Attributes:
        filename: the file who uses log
    """

    def __init__(self, filename):
        '''
        init
        :param filename:
        '''
        self.filename = os.path.basename(filename)
        if not os.path.exists('../logs'):
            os.makedirs("../logs")
        log_file = 'logs/%s.log' % self.filename
        handler = logging.handlers.RotatingFileHandler(log_file)  # 实例化handler
        formatter = logging.Formatter(LOG_FORMAT)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter
        logger = logging.getLogger(self.filename)  # 获取logger
        logger.addHandler(handler)  # 为logger添加handler
        logger.setLevel(LOG_LEVEL)

        self.logger = logger

    def get_logger(self):
        '''
        对外接口，获取logger
        :return: logger
        '''

        return self.logger