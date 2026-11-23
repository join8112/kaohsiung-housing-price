-- Monthly deals and median unit price for small (<= 2 rooms or studio) vs. family-size (3+ rooms)
-- apartments.  Used as the descriptive input of the credit-control policy study.
WITH seg AS (
    SELECT ym, unit_price,
           CASE WHEN btype = '套房' OR rooms BETWEEN 1 AND 2 THEN 'small (<=2 rooms)'
                WHEN rooms >= 3 THEN 'family (3+ rooms)'
           END AS segment
    FROM transactions
    WHERE btype IN ('大樓', '華廈', '公寓', '套房')
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY segment, ym ORDER BY unit_price) AS rn,
           COUNT(*)     OVER (PARTITION BY segment, ym)                     AS cnt
    FROM seg
    WHERE segment IS NOT NULL
)
SELECT segment, ym, MAX(cnt) AS deals, ROUND(AVG(unit_price), 2) AS median_unit_price
FROM ranked
WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
GROUP BY segment, ym
ORDER BY segment, ym;
