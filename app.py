#!flask/bin/python
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

