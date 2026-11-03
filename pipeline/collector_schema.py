"""수집기(../collector.py)와 같은 readings 표 구조. 재생기가 같은 모양으로 쓰기 위해 둔다."""
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS readings (
    ts_recv   INTEGER NOT NULL,
    ts_sensor INTEGER,
    source    TEXT NOT NULL,
    location  TEXT NOT NULL,
    spot      TEXT,
    device    TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    flag      INTEGER NOT NULL DEFAULT 0,
    msg_id    TEXT NOT NULL,
    UNIQUE (msg_id, device, metric)
);
CREATE INDEX IF NOT EXISTS idx_loc_metric_ts ON readings(location, metric, ts_recv);
CREATE INDEX IF NOT EXISTS idx_sensor_ts ON readings(ts_sensor);
"""
