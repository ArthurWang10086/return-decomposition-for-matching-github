#!/usr/bin/python
# -*- coding:utf-8 -*-
import pyhs2
import datetime
import os
import logging
import logging.handlers

DATE='2018-11-05'
LOG_FILE = 'get_new_roleids.log'          # 日志文件
SCRIPT_FILE = 'get_new_roleids'  # 脚本文件
LOG_LEVEL = logging.INFO                                                    # 日志级别
LOG_FORMAT = '%(asctime)s:%(lineno)s:%(levelname)s - %(message)s'    # 日志格式
def init_log():
    logger = logging.getLogger(SCRIPT_FILE)                          # 获取logger
    fmt = '%(asctime)s - %(funcName)s:%(lineno)s - %(message)s'      # 实例化formatter
    formatter = logging.Formatter(fmt)
    handler = logging.handlers.RotatingFileHandler(LOG_FILE)    # 实例化handler
    handler.setFormatter(formatter)                                              # 为handler添加formatter
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)                                                 # 为logger添加handler
    # 同时打印到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.info("log init")
    return logger

# GAME_RECORD_SQL = \
# '''
# SELECT concat(ds,'|',create_time,'|',server,'|',role_id) from balldb.ods_createrole where ds='{}'
# '''
GAME_RECORD_SQL = \
    '''
    select concat_ws(',',collect_set(cast(role_id as string))) from balldb.ods_gameend where ds='%s' and length(role_id)>5 group by game_uuid  having count(*)=6
    '''%(DATE)


class HiveClient:
    def __init__(self, db_host, user, password, database, port=10000, authMechanism="PLAIN", logger=None):
        """
        create connection to hive server2
        """
        self.conn = pyhs2.connect(host=db_host,
                                  port=port,
                                  authMechanism=authMechanism,
                                  user=user,
                                  password=password,
                                  database=database,
                                  )
        self.logger = logger

    def query(self, sql):

        """
        query
        """

        with self.conn.cursor() as cursor:

            cursor.execute(sql)

            return cursor.fetch()

    def action(self, sql):

        """
        query
        """

        with self.conn.cursor() as cursor:

            cursor.execute(sql)

            return cursor

    def pull_data(self, sql, out_file, sep="||"):
        with open(out_file, 'w') as fout, self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                _cnt = 0
                while cursor.hasMoreRows:
                    e = cursor.fetchone()
                    if e:
                        fout.write(sep.join(map(str, e)) + '\n')
                        _cnt += 1
                        if _cnt % 10000 == 0:
                            self.logger.debug("fetch %s %s: %s", out_file,  _cnt, ",".join(map(str, e[:3])))
                self.logger.info("pull data done. %s lines: %s", out_file, _cnt)
            except Exception as msg:
                self.logger.error("pull data failed: %s", msg)
            finally:
                pass
                # self.close()

    def close(self):

        """
        close connection
        """
        self.conn.close()


if __name__ == '__main__':
    logger = init_log()
    hive_client = HiveClient(db_host='59.111.7.43',
                             port=10000,
                             user='hdfs',
                             password='mypass',
                             database='default',
                             authMechanism='PLAIN',
                             logger=logger)
    SAVE_FOLDER = "new_role_ids/"
    if not os.path.exists(SAVE_FOLDER):
        logger.info("folder {} doesn't exist. Create folder.".format(SAVE_FOLDER))
        os.makedirs(SAVE_FOLDER)
    start_time = datetime.datetime.strptime(DATE, '%Y-%m-%d')
    delta_1 = datetime.timedelta(days=1)
    end_time = start_time
    logger.info("check game record files from {} to {}(included).".format(start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d')))
    for i in range((end_time-start_time).days+1):
        check_date = start_time + datetime.timedelta(days=i)
        game_record_file = SAVE_FOLDER + '/' + check_date.strftime('%Y-%m-%d') + ".csv"
        if os.path.exists(game_record_file):
            logger.info("game_record_file ({}) already exists, skip.".format(game_record_file))
        else:
            logger.info("game_record_file ({}) doesn't exist, start to download.".format(game_record_file))
            game_record_sql = GAME_RECORD_SQL.format(check_date.strftime('%Y-%m-%d'))
            print(game_record_sql)
            hive_client.pull_data(game_record_sql, game_record_file)
    hive_client.close()
    print('done')