-- District snapshot: last 12 months vs the 12 months before, with YoY change of the median.
WITH bounds AS (
    SELECT date(MAX(trade_date), 'start of month', '+1 month')  AS end_excl,
           date(MAX(trade_date), 'start of month', '-11 months') AS start_recent,
           date(MAX(trade_date), 'start of month', '-23 months') AS start_prev
    FROM transactions
),
labelled AS (
    SELECT t.district, t.unit_price, t.price_ex_park, t.age_years,
           CASE WHEN t.trade_date >= b.start_recent THEN 'recent' ELSE 'prev' END AS period
    FROM transactions AS t, bounds AS b
    WHERE t.trade_date >= b.start_prev AND t.trade_date < b.end_excl
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY district, period ORDER BY unit_price) AS rn,
           COUNT(*)     OVER (PARTITION BY district, period)                     AS cnt
    FROM labelled
),
med AS (
    SELECT district, period, AVG(unit_price) AS median_unit_price
    FROM ranked
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY district, period
),
recent AS (
    SELECT district,
           COUNT(*)                       AS deals_last_12m,
           ROUND(AVG(price_ex_park) / 1e4) AS mean_total_price_wan,
           ROUND(AVG(age_years), 1)        AS mean_age_years
    FROM labelled
    WHERE period = 'recent'
    GROUP BY district
)
SELECT r.district,
       r.deals_last_12m,
       ROUND(mr.median_unit_price, 2)                                   AS median_unit_price_last_12m,
       ROUND(mp.median_unit_price, 2)                                   AS median_unit_price_prev_12m,
       ROUND(100.0 * (mr.median_unit_price / mp.median_unit_price - 1), 1) AS yoy_pct,
       r.mean_total_price_wan,
       r.mean_age_years
FROM recent AS r
JOIN med AS mr ON mr.district = r.district AND mr.period = 'recent'
LEFT JOIN med AS mp ON mp.district = r.district AND mp.period = 'prev'
WHERE r.deals_last_12m >= 30
ORDER BY median_unit_price_last_12m DESC;
