SELECT
  d.day,
  SUM(f.amount) AS daily_sales
FROM
  payment_fact f
JOIN time_dim d ON DATE_TRUNC('day', f.payment_date) = d.payment_date
GROUP BY
  d.day
ORDER BY
  d.day;
