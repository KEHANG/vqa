import logging

from vqasite import app

logger = logging.getLogger('werkzeug')
handler = logging.FileHandler('access.log')
logger.addHandler(handler)

app.run(host="0.0.0.0", port=80, debug=True)