from sanic import Sanic
from sanic.response import json
from graphrank import extract_keyphrases as kpe

app = Sanic()


@app.route('/keyphase_extraction', methods=["POST"])
async def extract_keyphrases(request):
    data = request.json
    output = kpe.get_pim_keyphrases(data)
    return json(output)
