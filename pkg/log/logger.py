import logging
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now

        if log_record.get('level'):
            log_record['level'] = log_record['level'].lower()
        else:
            log_record['level'] = record.levelname.lower()

    def process_log_record(self, log_record):
        log_record['ts'] = log_record.pop('timestamp', None)

        msg  = log_record.pop('message', None)
        if msg is not None:
            log_record['msg'] = msg

        return jsonlogger.JsonFormatter.process_log_record(self, log_record)

def setup_server_logger(debug=False, dest=sys.stdout):
    l = logging.getLogger()

    logHandler = logging.StreamHandler(dest)
    formatter = CustomJsonFormatter('(timestamp) (level) (filename) (lineno) (module) (message)')
    logHandler.setFormatter(formatter)
    l.addHandler(logHandler)

    if debug:
        l.setLevel("DEBUG")
    else:
        l.setLevel("INFO")
