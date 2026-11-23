-- Cleaned residential sales, one row per transaction (built by src/khp/database.py)
DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    serial            TEXT PRIMARY KEY,   -- 實價登錄 official serial number (編號)
    district          TEXT NOT NULL,      -- 鄉鎮市區
    road              TEXT,               -- road parsed from the (partly masked) address
    trade_date        TEXT NOT NULL,      -- ISO date
    ym                TEXT NOT NULL,      -- 'YYYY-MM'
    quarter           TEXT NOT NULL,      -- 'YYYYQn'
    year              INTEGER,
    btype             TEXT NOT NULL,      -- 大樓 / 華廈 / 公寓 / 透天厝 / 套房
    total_price       INTEGER,            -- NT$
    price_ex_park     INTEGER,            -- NT$, separately priced parking removed
    unit_price        REAL,               -- 萬元 per 坪, excluding parking
    area_ex_park_ping REAL,
    land_area_ping    REAL,
    age_years         REAL,
    floor             REAL,
    total_floors      REAL,
    rooms             INTEGER,
    halls             INTEGER,
    baths             INTEGER,
    elevator          INTEGER,
    management        INTEGER,
    has_parking       INTEGER,
    parking_unpriced  INTEGER,
    parking_price     INTEGER
);
