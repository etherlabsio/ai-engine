from sanic import Sanic
from sanic.response import json
import json as js
from community import pims, communities
from community.transport import decode_json_request
from log.logger import setup_server_logger
from community import loadmodel
import logging

logger = logging.getLogger()

bert_config = {}
bert_config["tokenizer"] = 'bert-base-uncased'
bert_config["config"] = 'mind_files/bert_config.json'
bert_config["bert_model"] = 'bert-base-uncased'
bert_config["load_file"] = "mind_files/"


if __name__ == '__main__':
        setup_server_logger(debug=True)
        app = Sanic()
        @app.route('/calculate_pims', methods=["POST"])
        async def identifyPIMs(request):
                segments = decode_json_request(request.json)
                mind = loadmodel.selectmodel((request.json)['mindId'])
                model1 = loadmodel.loadmodel(bert_config, mind)
                res = pims.computepims(segments, model1)
                #reqData = request.json
                #res = getpims.computerpims(reqData)
                return json(res)
        @app.route('/debug/healthcheck')
        async def healthcheck(request):
                return json({"message": "healthy"})
        app.run(host='0.0.0.0', port=8080)
