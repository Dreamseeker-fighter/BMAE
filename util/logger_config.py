#!/usr/bin/env python3
# encoding: utf-8


import os
from loguru import logger
from cmreslogging.handlers import CMRESHandler
from elasticsearch import Elasticsearch
es = Elasticsearch(hosts="10.0.0.145:32321")
author =  os.getlogin()
handler = CMRESHandler(hosts=[{'host': '10.0.0.145', 'port': 32321}],
                           auth_type=CMRESHandler.AuthType.NO_AUTH,
                           es_index_name="anomaly_detection_log",
                           es_additional_fields={'author': author,

                                                 }
)

logger.add(handler)
# logger.add("{}.log".format(author))

