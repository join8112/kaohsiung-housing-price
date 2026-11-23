CREATE INDEX idx_tx_district_ym ON transactions (district, ym);
CREATE INDEX idx_tx_ym          ON transactions (ym);
CREATE INDEX idx_tx_btype       ON transactions (btype);
CREATE INDEX idx_tx_road        ON transactions (district, road);
