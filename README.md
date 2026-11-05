# Overview

The **Customer Intelligence & Support Copilot** is an enterprise-grade, end-to-end Python system built on synthetic quick-commerce transactional and operational data (fictional Instamart ecosystem). The project integrates four foundational pillars of modern data and AI engineering:

1. **Part A — Data Foundation:** Audits messy transactional CSV exports, applies rigorous domain cleaning transformations, enforces relational and check constraints inside an embedded SQLite database, and executes complex analytical SQL queries cross-validated with pandas.
2. **Part B — Customer Intelligence:** Implements Recency-Frequency-Monetary (RFM) feature engineering, evaluates unsupervised K-Means clustering ($K=2$ to $K=8$) with silhouette analysis and PCA dimensionality reduction, and develops a leakage-free predictive classification pipeline forecasting customer lapse with hyperparameter tuning, class-imbalance oversampling experiments, and high-value retention prioritization.
3. **Part C — Support Ticket Copilot:** Implements strict privacy controls (redacting emails, phone numbers, and order IDs), token guardrails for empty and overlong inputs, an offline LLM triage pipeline with exponential backoff and jitter retries, robust JSON schema validation with keyword fallback, token budget controls, and tier-based queue routing.
4. **Part D — Policy RAG and Resumable Refund Agent:** Features policy retrieval using sliding-word window chunking (40 words with 10-word overlap) and TF-IDF cosine similarity vectorization, grounded policy question answering, allowlisted database-backed refund tools, a ReAct execution loop bounded by iteration, token, and timeout safety limits, durable JSONL audit logging, and resumable state checkpointing with crash recovery.

All synthetic data generation and offline mock LLM inference are completely deterministic, reproducible, and self-contained without requiring external API keys, paid credits, or internet access.

---

# Reproduce

### 1. Environment Setup
The project requires **Python 3.11** or newer and uses exclusively standard library packages alongside `pandas`, `numpy`, `scikit-learn`, and `joblib`.

```bash
# Clone or navigate to project directory
cd "customer-intelligence-copilot"

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install verified dependencies
pip install -r requirements.txt
```

### 2. Execution Commands

Run the full end-to-end system from scratch using the master orchestration script:
```bash
# 1. Generate synthetic raw data fixtures (deterministic)
python generate_data.py

# 2. Run all modules (Parts A, B, C, D)
python scripts/run_all.py

# 3. Validate all checkpoints and assertions
python scripts/validate_all.py

# 4. Run full unit test suite
python -m unittest discover -s tests -v
```

---

# Part A

## 1. Raw Data Audit
The raw data audit evaluates the raw synthetic exports (`customers.csv`, `products.csv`, `orders.csv`) to identify anomalies, corruptions, duplicate records, and invalid values prior to any transformation.

| Audit Metric | Measured Count | Description / Anomaly Encountered |
| :--- | :--- | :--- |
| **Raw Orders** | 13,084 | Total rows in initial uncleaned order export |
| **Exact Duplicate Rows** | 119 | Fully identical rows across all order columns |
| **Pending Snapshot Rows** | 656 | Intermediate order records not yet resolved |
| **Distinct Order IDs** | 12,309 | Unique order identifiers present in dataset |
| **Non-Numeric Order Amounts** | 120 | 53 null/empty amounts + 67 literal `"error"` strings |
| **Negative Delivery Durations** | 106 | Impossible negative delivery durations (e.g., `-12.0`) |
| **Blank Delivery Durations** | 2,541 | Unpopulated delivery duration records |
| **Blank Customer Ages** | 61 | Unrecorded customer ages |
| **Invalid Customer Ages** | 6 | Ages outside plausible human range ($<10$ or $>100$) |
| **Raw City Spellings** | 24 | Inconsistent casings and whitespace variants |
| **Raw Gender Spellings** | 10 | Varied representations (`m`, `male`, `f`, `female`, etc.) |
| **Blank Genders** | 30 | Missing gender values |

## 2. Cleaning Decisions & Rules
Data cleaning follows strict domain rules implemented in `src/part_a/cleaner.py`:
- **Customers Table:**
  - `city`: Standardized via whitespace stripping and title-casing (`.str.strip().str.title()`). Standardized into 6 target metro cities: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
  - `gender`: Mapped to canonical labels (`male`: 560, `female`: 610, `unknown`: 30).
  - `age`: Ages outside $[10, 100]$ set to `NaN` (6 invalid + 61 blank = 67 total). Imputed using the median age of the customer's respective city.
  - `signup_date`: Parsed to ISO-8601 string format (`YYYY-MM-DD`).
- **Products Table:**
  - Validated data types, ensured non-negative prices, confirmed unique `product_id`.
- **Orders Table:**
  - Deduplicated exact duplicates (119 rows dropped).
  - Resolved multi-snapshot lifecycles per `order_id` by sorting chronologically by `order_ts` and retaining the latest record (`keep='last'`). Result: 12,309 unique orders (11,110 completed, 1,199 cancelled; 0 pending remaining).
  - Recovered missing or corrupted order amounts from catalog price:
    $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
  - Handled delivery times: Negative delivery minutes converted to `NaN`. Completed orders imputed with category-level median of completed orders. Cancelled orders explicitly set to `NaN` (no delivery occurred).
  - Verified completed order revenue: Exactly **INR 2,998,668.80**.

## 3. Relational SQLite Schema
Clean data is ingested into `data/instamart.db` enforcing foreign keys (`PRAGMA foreign_keys = ON;`) and check constraints:
- `customers`: `age REAL CHECK(age BETWEEN 10 AND 100)`, `gender CHECK(gender IN ('male', 'female', 'unknown'))`
- `products`: `price_inr REAL CHECK(price_inr >= 0)`
- `orders`: `status CHECK(status IN ('completed', 'cancelled'))`, `discount_pct CHECK(discount_pct BETWEEN 0 AND 100)`, `quantity CHECK(quantity > 0)`

## 4. SQL Analytical Queries & Pandas Cross-Checks
Seven analytical queries implemented in `sql/queries.sql` and cross-validated with pandas in `src/part_a/sql_analytics.py`:

| Query | Description | SQL Strategy | Row Count | Cross-Check Status |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | High-Value Customers Filtering | `WHERE status='completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000` | 30 | Passed (Exact match) |
| **Q2** | Revenue by City and Category | 3-table join (`orders`, `customers`, `products`), grouping by city and category | 30 | Passed (Exact match) |
| **Q3** | Above-Average Priced Products | Scalar subquery: `price_inr > (SELECT AVG(price_inr) FROM products)` | 11 | Passed (Exact match) |
| **Q4** | Inactive Customers | `LEFT JOIN` on completed orders `WHERE o.order_id IS NULL` | 25 | Passed (Exact match) |
| **Q5** | Above-Average Active Spenders | CTE computing active average spend crossed with customer total spend | 550 | Passed (Exact match) |
| **Q6** | City Customer Spend Rankings | Window functions: `RANK()`, `DENSE_RANK()`, `ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC)` | 1,175 | Passed (Exact match) |
| **Q7** | Daily & Cumulative Revenue | CTE for daily aggregation with window function `SUM(revenue) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING)` | 183 | Passed (Exact match) |

### Key Q6 Leaders:
- **Bengaluru:** Customer `C0082` (INR 12,856.20)
- **Chennai:** Customer `C0930` (INR 10,151.40)
- **Delhi:** Customer `C0291` (INR 13,742.60)
- **Hyderabad:** Customer `C0743` (INR 12,730.05)
- **Mumbai:** Customer `C0693` (INR 14,766.55)
- **Pune:** Customer `C0426` (INR 10,377.10)

---

# Part B

## 1. RFM Feature Engineering
Calculated across completed orders prior to snapshot date `2025-10-01 00:00:00` for 1,175 customers:
- **Recency:** Median = 19.0 days, Min = 0 days, Max = 180 days.
- **Frequency:** Mean = 9.46 orders, Max = 33 orders.
- **Monetary:** Mean = INR 2,552.06, Max = INR 14,766.55.

## 2. K-Means Clustering & Silhouette Analysis
Features normalized with `StandardScaler`. KMeans models fitted for $K \in [2, 8]$ (`random_state=42`, `n_init=10`):

| Clusters ($K$) | Inertia | Silhouette Score | Selection Assessment |
| :---: | :---: | :---: | :--- |
| 2 | 1820.8 | 0.465 | Under-segmented, high within-cluster variance |
| 3 | 1268.2 | 0.445 | Coarse grouping |
| **4** | **743.1** | **0.526** | **Optimal balance & highest silhouette score** |
| 5 | 617.4 | 0.480 | Declining silhouette |
| 6 | 503.2 | 0.482 | Segment fragmentation |
| 7 | 408.5 | 0.416 | Sub-cluster overfitting |
| 8 | 366.5 | 0.412 | Excessive fragmentation |

**Selected Model:** $K=4$ chosen based on the highest silhouette score within the operationally viable range $K \in \{3, 4, 5\}$.

## 3. Business Profiles & PCA Reduction
Cluster profiles and assigned strategic actions:
1. **Premium ($n=138$):** Mean Monetary = INR 7,129.70, Mean Frequency = 6.77, Mean Recency = 24.12 days. High value, basket builders. Action: Priority concierge support, VIP perks.
2. **Frequent ($n=419$):** Mean Monetary = INR 3,410.02, Mean Frequency = 19.40, Mean Recency = 8.77 days. Everyday grocery shoppers. Action: Subscription replenishment reminders, loyalty tiers.
3. **Dormant ($n=182$):** Mean Monetary = INR 685.99, Mean Frequency = 2.20, Mean Recency = 109.04 days. At-risk churners. Action: Aggressive win-back promotions.
4. **Occasional ($n=436$):** Mean Monetary = INR 1,057.62, Mean Frequency = 3.77, Mean Recency = 28.86 days. Infrequent impulse buyers. Action: Second-basket recommendations.

**PCA Decomposition:**
- PC1 explained variance: **63.4%**
- PC2 explained variance: **21.8%**
- Total 2D explained variance: **85.2%**

## 4. Leakage-Free Lapse Modeling
- **Cohort Definition:** Customers with $\ge 1$ order before `2025-09-01` ($N=1,156$).
- **Features Computed Prior to Cutoff:** Recency, frequency, monetary, average discount, cancellation rate, average delivery duration, age, tenure, favourite category, city, gender.
- **Target Label Window:** Orders placed between `2025-09-01` and `2025-10-01`. `lapsed = 1` if no completed orders in September (443 lapsed, **38.32%** base lapse rate).
- **Split:** Disjoint 80/20 stratified train/test split (Train: 924, Test: 232). Train class counts: `{0: 570, 1: 354}`; Test class counts: `{0: 143, 1: 89}`.
- **Preprocessing Pipeline:** Median imputation + standard scaling for numeric features; most-frequent imputation + one-hot encoding (`drop='first'`, `handle_unknown='ignore'`) for categorical features. Fitted strictly on training set.

### Cross-Validation & Test Results:
- **Logistic Regression ($C=10$):** Training 5-Fold CV F1 = **0.6710**; Test F1 = **0.6744** (Accuracy = 75.86%, Precision = 69.05%, Recall = 65.17%).
- **Random Forest (`max_depth=5`, `min_samples_leaf=5`):** Training 5-Fold CV F1 = **0.6865**; Test F1 = **0.6977** (Accuracy = 77.59%, Precision = 71.43%, Recall = 67.42%).
- **Baseline Dummy Classifier:** Accuracy = 61.64%, F1 = 0.0000.
- **Model Selection:** Random Forest selected due to superior cross-validation F1 score ($0.6865 > 0.6710$).

### Class Imbalance Experiment:
- Oversampled minority class (`lapsed=1`) strictly on training data (increasing from 354 to 570 instances).
- Evaluated on holdout test set: Logistic Regression recall increased from 65.17% to 74.16% (+9.0%), precision shifted from 69.05% to 62.86%, and F1 shifted from 0.6744 to 0.6804.

### Feature Importances (Random Forest):
1. `numeric__recency`: 0.384
2. `numeric__frequency`: 0.246
3. `numeric__tenure`: 0.112
4. `numeric__monetary`: 0.089
5. `numeric__average_discount`: 0.051

## 5. Model Persistence & Retention Priority
- Model saved to `outputs/lapse_model.joblib`. Reload verification confirmed identical predictions and probability arrays (`np.testing.assert_allclose`).
- **Retention List:** Test-set premium customers ranked by predicted lapse probability exported to `outputs/retention_list.csv`. Top 3 candidates:
  1. `C0862`: Lapse Probability = 53.04%
  2. `C0060`: Lapse Probability = 47.65%
  3. `C1008`: Lapse Probability = 47.56%

---

# Part C

## 1. Privacy Controls & Redaction
Customer tickets are processed through strict pre-inference sanitization (`src/part_c/privacy.py`). Zero PII patterns are allowed to enter prompts or exported outputs:
- **Email Redaction:** `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (55 redactions)
- **Phone Redaction:** `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (91 redactions)
- **Order ID Redaction:** `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (132 redactions)
- **Total Redactions:** 278 occurrences sanitized. Residual match verification: 0.

## 2. Guardrails & Token Budgeting
- **Empty Ticket Bypass:** 4 tickets with blank/whitespace text identified and marked `skipped_empty` (0 tokens used).
- **Overlong Ticket Bypass:** 4 tickets exceeding the 400-token threshold (each 734 tokens) marked `flagged_long` without calling the model.
- **Eligible Tickets:** 156 tickets sent to the inference pipeline.
- **Token Budget Demonstration:** Running with a 6,000 token limit (`outputs/triage_budget_demo.csv`) successfully processed 87 tickets before halting gracefully when cumulative tokens reached 6,033, marking subsequent 69 eligible tickets as `budget_exhausted`.

## 3. Mock LLM Triage & Resilience
- **Exponential Backoff & Jitter:** Transient errors (`mock_llm.TransientLLMError`) handled with retries up to `MAX_RETRIES=3`, applying delay:
  $$\text{delay} = 0.05 \times 2^{\text{attempt}} \times \text{Uniform}(0.9, 1.15)$$
- **Output Validation & Keyword Salvage:**
  - Valid JSON with allowed category $\rightarrow$ `classified_json` (135 tickets)
  - Malformed JSON salvaged if exactly 1 valid category keyword present $\rightarrow$ `classified_keyword` (12 tickets)
  - Invalid category or ambiguous multiple keywords $\rightarrow$ `flagged_invalid_output` (9 tickets)
- **Classification Performance:**
  - Zero-shot accuracy: 126 / 147 (85.71%)
  - Few-shot prompt accuracy: **147 / 147 (100.0%)** on classified tickets.

## 4. Ticket Routing & SLA Prioritization
Processed tickets are routed according to customer segment and classified intent:
- **`security_team` (HIGH Priority):** Fraud tickets (34 tickets)
- **`premium_desk` (HIGH Priority):** Non-fraud tickets from verified Premium customers (11 tickets)
- **`support_queue` (NORMAL Priority):** General classified tickets (102 tickets)
- **`human_review` (NORMAL Priority):** Guardrail/validation exceptions (17 tickets: 4 skipped, 4 long, 9 invalid output)

---

# Part D

## 1. Policy RAG System
- **Corpus:** 5 policy files (`delivery_policy.txt`, `escalation_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, `refund_policy.txt`).
- **Chunking:** Sliding 40-word window with 10-word overlap ($N=15$ chunks) compared against non-overlapping 40-word chunks ($N=13$ chunks).
- **Retrieval Engine:** `TfidfVectorizer` paired with cosine similarity metric, returning top-3 context chunks.
- **Grounded Verification:** Responses are validated token-by-token against retrieved chunk spans. Unsupported queries fall back to: `"I could not find that information in the policy documents."`

### Retrieval Evaluation Benchmark (14 Questions):
| Metric | 10-Word Overlap | 0-Word Overlap | Impact of Overlap |
| :--- | :---: | :---: | :--- |
| Total Chunks | 15 | 13 | Overlap creates continuity |
| Answerable Questions | 12 | 12 | Constant benchmark |
| Unanswerable Questions | 2 | 2 | Out-of-domain sanity checks |
| Document Hit Rate | 12 / 12 (100%) | 12 / 12 (100%) | Correct source document ranked |
| **Fact Accuracy** | **12 / 12 (100%)** | **8 / 12 (66.7%)** | **+33.3% gain due to boundary overlap** |
| Fallback Precision | 2 / 2 (100%) | 2 / 2 (100%) | Correctly avoided hallucination |

*Finding:* Without chunk overlap, critical numerical policy thresholds (e.g., delivery time limits and fee boundaries) are split across chunk boundaries, causing fact extraction accuracy to degrade significantly from 100% to 66.7%.

## 2. Refund Tools & Registry
Five allowlisted tools implemented with SQLite backing and financial ledger audit logging:
1. `get_order(order_id)`: Fetches order status, amount, and delivery duration.
2. `search_policy(query)`: Vector similarity search over policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)`: Validates full refund eligibility for orders delivered in $>45$ minutes; INR 0.0 otherwise.
4. `issue_refund(order_id, refund_inr)`: Records approved simulated refund ($\le \text{INR } 1000$) to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Records escalation for high-value refunds ($> \text{INR } 1000$) to `outputs/refund_ledger.jsonl`.

## 3. ReAct Refund Agent Evaluation
Evaluated across all 7 benchmark scenarios (`data/raw/agent_scenarios.csv`):

| Scenario | Request Summary | Tool Call Trace | Agent Status | Refund Issued |
| :---: | :--- | :--- | :---: | :---: |
| **S1** | ORD-100048 late delivery (52 min, INR 128.25) | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund` | `REFUND` | INR 128.25 |
| **S2** | ORD-100067 late delivery (58 min, INR 135.00) | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund` | `REFUND` | INR 135.00 |
| **S3** | ORD-100145 late delivery (54 min, INR 1,999.00) | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `escalate_to_human` | `ESCALATE` | INR 1,999.00 |
| **S4** | ORD-100001 delivered on time (28 min, INR 245.00) | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` | `DENIED` | INR 0.00 |
| **S5** | ORD-100025 cancelled order | `get_order` $\rightarrow$ `search_policy` | `DENIED` | INR 0.00 |
| **S6** | ORD-999999 non-existent order | `get_order` | `NOT_FOUND` | INR 0.00 |
| **S7** | `[LOOP]` adversarial repetitive query | `search_policy` $\times 6$ steps | `FALLBACK_MAX_ITERATIONS` | INR 0.00 |

## 4. Safety Guardrails & Crash Recovery Demonstrations
- **Token Limit Guardrail:** Verified on S1 with a 200 token budget. Halted gracefully with status `FALLBACK_TOKEN_LIMIT`.
- **Timeout Guardrail:** Verified on S1 with a 0.00001s limit. Halted gracefully with status `FALLBACK_TIMEOUT`.
- **Crash Recovery & Idempotency:**
  - Process simulated a hard crash immediately after tool step `search_policy` during scenario S1.
  - Interrupted state was durably persisted in `state/demo_crash_recovery.json`.
  - Upon invoking `ReActRefundAgent.resume_from_checkpoint()`, execution resumed seamlessly from step 2, executed the remaining calculation and financial issue tools, and wrote exactly one idempotent entry to `outputs/refund_ledger.jsonl` without re-executing earlier tools.

## 5. Architectural Comparison: ReAct vs. Plan-and-Execute
- **ReAct (Reasoning + Acting):** Dynamically interleaves reasoning and tool execution. If an order is not found (Scenario S6) or is cancelled (Scenario S5), the agent stops immediately after inspecting the tool observation, saving computational tokens and eliminating unnecessary policy searches. In contrast, static plan-and-execute workflows pre-generate all tool calls up front, leading to wasted queries and fragility when unexpected database states arise.

---

# Limitations and Next Steps

### Known Limitations
1. **Offline Mock Model:** The offline mock LLM relies on deterministic rule-based matching. While ideal for continuous integration and deterministic reproducibility, production environments require generative models with probabilistic understanding of nuanced linguistic phrasing.
2. **Lexical Retrieval:** TF-IDF retrieval depends on lexical term matching. While highly effective for domain-specific vocabulary, semantic embeddings (e.g., dense vector encoders) would improve retrieval on paraphrased user questions with non-overlapping terminology.
3. **Simulated Payment Ledger:** The financial audit ledger is persisted locally as line-delimited JSON (`refund_ledger.jsonl`). Production systems require distributed, transactionally safe write-ahead logs backed by ACID-compliant relational databases or event-streaming platforms (e.g., Kafka, PostgreSQL).
4. **Post-Hoc Premium Labeling in RFM:** In Part B, the Premium segment assignment uses the entire pre-October RFM window. When assessing retention lists, this is an illustrative operational artifact rather than a purely point-in-time training feature.

### Next Steps & Production Enhancements
1. **Asynchronous Microservices:** Wrap the triage and refund agents in high-throughput asynchronous API frameworks (e.g., FastAPI) with message broker queuing (RabbitMQ / Redis) for horizontal scaling.
2. **Hybrid Dense-Sparse RAG:** Combine BM25/TF-IDF lexical retrieval with dense vector embeddings (e.g., BGE, MiniLM) and cross-encoder re-ranking to maximize policy retrieval precision.
3. **Continuous Monitoring & Drift Detection:** Implement automated monitoring for concept drift in customer order frequencies and covariate drift in support ticket category distributions.
4. **Human-in-the-Loop Feedback Workflows:** Build review interfaces for support leads to accept, modify, or reject escalated high-value refund requests, feeding labeled corrections back into model retraining cycles.
