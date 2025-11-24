from __future__ import annotations

import threading
from typing import Optional

from clickhouse_connect import get_client
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


# Thread-local storage: кожен потік матиме свій client
_client_local = threading.local()


def get_ch_client(settings: Optional[ClickHouseSettings] = None):
    client = getattr(_client_local, "client", None)
    if client is not None:
        return client

    s = settings or ClickHouseSettings()
    client = get_client(
        host=s.host,
        port=s.port,
        username=s.username,
        password=s.password,
        database=s.database,
        connect_timeout=s.connect_timeout,
        send_receive_timeout=s.send_receive_timeout,
    )
    # sanity ping
    _ = client.query("SELECT 1").first_item

    _client_local.client = client
    return client
