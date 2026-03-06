import json
import os
from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()

redis = Redis(
    url=os.getenv("REDIS_URL"),
    token=os.getenv("REDIS_TOKEN")
)


def save_session(session_id, data):    
    key = f"session:{session_id}"
    redis.set(key, json.dumps(data), ex=86400)  # 24h expiry


def get_session(session_id):
    key = f"session:{session_id}"
    data = redis.get(key)

    if data:
        return json.loads(data)

    return None


def update_session_field(session_id, field, value):
    session = get_session(session_id) or {}
    session[field] = value
    save_session(session_id, session)