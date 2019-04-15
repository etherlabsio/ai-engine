from sanic import Sanic
from sanic.response import json
from graphrank import extract_keyphrases as kpe

app = Sanic()


@app.route('/keyphase_extraction.graph', methods=["POST"])
async def identifyPIMs(request):
    data = request.json
    modified_input_req = kpe.compute_keyphrases(data)
    return json(modified_input_req)
