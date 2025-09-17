from __future__ import annotations

import threading
from typing import Optional

from clickhouse_connect import common, get_client
from pydantic import BaseModel


class ClickHouseSettings(BaseModel):
    host: str = "localhost"
    port: int = 8123  # HTTP port
    username: str = "default"
    password: str = "1234"
    database: str = "codebase"
    # optional tuning
    connect_timeout: int = 10
    send_receive_timeout: int = 60


_client = None
_lock = threading.Lock()


def get_ch_client(settings: Optional[ClickHouseSettings] = None):
    """
    Thread-safe, process-local singleton client.
    """
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            s = settings or ClickHouseSettings()
            _client = get_client(
                host=s.host,
                port=s.port,
                username=s.username,
                password=s.password,
                database=s.database,
                connect_timeout=s.connect_timeout,
                send_receive_timeout=s.send_receive_timeout,
            )
            # sanity ping
            _ = _client.query("SELECT 1").first_item
    return _client
