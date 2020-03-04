import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Mapping
from walrus import Database, Hash
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(order=True)
class RedisObject:
    host: str = field(default="localhost")
    id: str = field(default=None, compare=True)
    port: int = 6379
    decode_responses: bool = True
    redis: Database = None

    @staticmethod
    def encode_value(value):

        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.warning(e)
            raise

    @staticmethod
    def decode_value(value):

        try:
            if value is not None:
                return pickle.loads(value)
            else:
                return None
        except Exception as e:
            logger.warning(e)
            raise


@dataclass
class RedisStore(RedisObject):
    id: str
    host: str
    h: Hash = field(default=None)

    def __post_init__(self):
        self.redis = Database(host=self.host, port=self.port, db=0)
        self.h = self.redis.Hash(key=self.id)

    def set_object(self, key, object: Any):
        self.h.update({key: RedisObject.encode_value(object)})

        logger.info("set object in redis store")

    def get_object(self, key):
        obj = RedisObject.decode_value(self.h.get(key=key))

        logger.info("retrieved object from redis store")
        return obj

    def delete_key(self, key):
        try:
            del self.h[key]
            return True
        except Exception as e:
            logger.warning(e)
            return False

    def expire_store(self, time):
        self.h.expire(ttl=time)

    def get_dict(self):
        return self.h.as_dict()
