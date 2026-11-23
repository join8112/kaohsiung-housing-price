-- District x quarter median unit price, with year-over-year change via LAG().
WITH ranked AS (
    SELECT district, quarter, unit_price,
           ROW_NUMBER() OVER (PARTITION BY district, quarter ORDER BY unit_price) AS rn,
           COUNT(*)     OVER (PARTITION BY district, quarter)                     AS cnt
    FROM transactions
),
q AS (
    SELECT district, quarter, MAX(cnt) AS deals, AVG(unit_price) AS median_unit_price
    FROM ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY district, quarter
)
SELECT *
FROM (
    SELECT district,
           quarter,
           deals,
           ROUND(median_unit_price, 2) AS median_unit_price,
           ROUND(100.0 * (median_unit_price
                 / LAG(median_unit_price, 4) OVER (PARTITION BY district ORDER BY quarter) - 1), 1) AS yoy_pct
    FROM q
)
WHERE deals >= 20   -- filter after LAG() so the 4-quarter offset is not shifted
ORDER BY district, quarter;
