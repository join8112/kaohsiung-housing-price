-- Most expensive roads in each district over the last 12 months (roads with >= 20 deals).
WITH recent AS (
    SELECT *
    FROM transactions
    WHERE trade_date >= (SELECT date(MAX(trade_date), 'start of month', '-11 months') FROM transactions)
      AND road IS NOT NULL
),
road_stats AS (
    SELECT district, road,
           COUNT(*)                      AS deals,
           ROUND(AVG(unit_price), 2)     AS mean_unit_price,
           ROUND(AVG(age_years), 1)      AS mean_age_years
    FROM recent
    GROUP BY district, road
    HAVING COUNT(*) >= 20
)
SELECT *
FROM (
    SELECT *,
           DENSE_RANK() OVER (PARTITION BY district ORDER BY mean_unit_price DESC) AS rank_in_district
    FROM road_stats
)
WHERE rank_in_district <= 3
ORDER BY district, rank_in_district;
