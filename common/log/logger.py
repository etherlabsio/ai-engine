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
        if log_record.get('message'):
            log_record['message'] = log_record['message'] or log_record['msg']

def new_server_logger():
    l = logging.getLogger()

    logHandler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter('(timestamp) (level) (message)')
    logHandler.setFormatter(formatter)
    l.addHandler(logHandler)
    l.setLevel("INFO")
    return l


if __name__ == '__main__':
    l = new_server_logger()
    l.info({
        "message": "looks like it's going well"
    })

    logging.getLogger().info({
        "test":"me"
    })

    logging.getLogger().error("this is a debug test", extra={
        "test": 1234
    })
