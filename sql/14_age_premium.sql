-- How unit price falls with building age, by building type.
WITH bucketed AS (
    SELECT btype, unit_price,
           CASE
               WHEN age_years < 5  THEN '0-4'
               WHEN age_years < 10 THEN '5-9'
               WHEN age_years < 20 THEN '10-19'
               WHEN age_years < 30 THEN '20-29'
               WHEN age_years < 40 THEN '30-39'
               ELSE '40+'
           END AS age_bucket,
           CASE
               WHEN age_years < 5  THEN 1 WHEN age_years < 10 THEN 2
               WHEN age_years < 20 THEN 3 WHEN age_years < 30 THEN 4
               WHEN age_years < 40 THEN 5 ELSE 6
           END AS bucket_order
    FROM transactions
    WHERE age_years IS NOT NULL
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY btype, age_bucket ORDER BY unit_price) AS rn,
           COUNT(*)     OVER (PARTITION BY btype, age_bucket)                     AS cnt
    FROM bucketed
)
SELECT btype, age_bucket, MAX(cnt) AS deals, ROUND(AVG(unit_price), 2) AS median_unit_price
FROM ranked
WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
GROUP BY btype, age_bucket, bucket_order
HAVING MAX(cnt) >= 30
ORDER BY btype, bucket_order;
