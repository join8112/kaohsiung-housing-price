-- Citywide monthly volume and median unit price (SQLite has no MEDIAN(), so use window functions).
WITH ranked AS (
    SELECT ym,
           unit_price,
           ROW_NUMBER() OVER (PARTITION BY ym ORDER BY unit_price) AS rn,
           COUNT(*)     OVER (PARTITION BY ym)                     AS cnt
    FROM transactions
),
medians AS (
    SELECT ym, AVG(unit_price) AS median_unit_price
    FROM ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY ym
)
SELECT t.ym,
       COUNT(*)                                                    AS deals,
       ROUND(m.median_unit_price, 2)                               AS median_unit_price,
       ROUND(AVG(t.unit_price), 2)                                 AS mean_unit_price,
       ROUND(AVG(CASE WHEN t.btype = '透天厝' THEN 1.0 ELSE 0 END), 3) AS share_townhouse,
       ROUND(AVG(CASE WHEN t.age_years < 5 THEN 1.0 ELSE 0 END), 3)   AS share_age_under_5
FROM transactions AS t
JOIN medians AS m USING (ym)
GROUP BY t.ym
ORDER BY t.ym;
