-- ============================================================================
-- Instamart Customer Intelligence & Support Copilot
-- Analytical SQL Queries (Q1 - Q7)
-- ============================================================================

-- Q1: High-Value Customers Filtering
-- Identify customers whose qualifying spend exceeds INR 8,000,
-- considering only individual completed orders above INR 500.
SELECT 
    customer_id, 
    ROUND(SUM(order_amount), 2) AS qualifying_spend
FROM orders
WHERE status = 'completed' 
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC, customer_id;

-- Q2: Revenue by City and Product Category
-- Aggregate completed order revenue across customers and products.
SELECT 
    c.city, 
    p.category, 
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC, c.city, p.category;

-- Q3: Above-Average Priced Products
-- Select products priced above the overall catalog average.
SELECT 
    product_id, 
    product_name, 
    category, 
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4: Inactive Customers
-- Find customers who have never placed a completed order.
SELECT 
    c.customer_id, 
    c.city
FROM customers AS c
LEFT JOIN orders AS o 
  ON o.customer_id = c.customer_id 
 AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customers Spending Above Active Average
-- Identify customers whose total completed spend exceeds the mean spend
-- across all customers who have placed at least one completed order.
WITH customer_spend AS (
    SELECT 
        customer_id, 
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend 
    FROM customer_spend
)
SELECT 
    c.customer_id, 
    c.total_spend, 
    a.avg_spend AS average_spend
FROM customer_spend AS c
CROSS JOIN average_spend AS a
WHERE c.total_spend > a.avg_spend
ORDER BY c.total_spend DESC, c.customer_id;

-- Q6: City-Wise Customer Spend Rankings
-- Rank customers within each city by their total completed spend.
WITH customer_spend AS (
    SELECT 
        c.city, 
        c.customer_id, 
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT 
    city, 
    customer_id, 
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number
FROM customer_spend
ORDER BY city, row_number;

-- Q7: Daily and Cumulative Completed Revenue
-- Calculate daily revenue and running cumulative revenue over time.
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS day,
        ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day, 
    revenue,
    ROUND(SUM(revenue) OVER (
        ORDER BY day 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
