#!/usr/bin/env python
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#

import json
import ast
import re
from collections import defaultdict
from flask import Flask, jsonify, send_from_directory, redirect
try:
    from urllib.request import Request, build_opener
    from urllib.parse import urlencode
except ImportError:
    from urllib2 import Request, build_opener
    from urllib import urlencode


@app.route('/app/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory('app', path)

@app.route('/app/', methods=['GET'])
def static_index():
    return send_from_directory('app', 'index.html')

