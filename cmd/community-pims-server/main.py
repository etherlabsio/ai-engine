from sanic import Sanic
from sanic.response import json
import json as js
from community import getpims, getcommunity
from log.logger import setup_server_logger
import logging
logger = logging.getLogger()

if __name__ == '__main__':
        setup_server_logger(debug=True)
        app = Sanic()
        @app.route('/channelminds.identifyPIMs/', methods=["POST"])
        async def identifyPIMs(request):
                reqData = request.json
                res = getpims.computerpims(reqData)
                return json(res)
        @app.route('/debug/healthcheck')
        async def healthcheck(request):
                return json({"message": "healthy"})
        app.run(host='0.0.0.0', port=8080)
