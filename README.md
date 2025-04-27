# EECS 4415 – Spark Assignments

This repository contains concise, expert-level solutions for six Spark-based tasks, showcasing text processing, pattern mining, clustering, and dimensionality reduction.

---

## Task 1: Word Count in Spark
Write a Spark application that:
1. Installs and configures PySpark in Colab.
2. Reads `pg100.txt` (complete works of Shakespeare).
3. Splits lines on spaces, normalizes to lowercase, and filters words starting with A–Z.
4. Maps each word to `(first_letter, 1)` and reduces by key to count total occurrences.
5. Outputs counts for the letters: `a, d, e, j, k, n, q, t, y, z`.
6. Generates a bar chart of these counts with pandas/matplotlib.
7. Concludes with a brief paragraph summarizing linguistic patterns and Spark’s efficiency.

## Task 2: Pattern Discovery in Spark
Using the Instacart subset (~131K orders):
1. Install PySpark and load `products.csv` & `order_products__train.csv` via PyDrive.
2. Join orders with products to form `baskets` (list of names per `order_id`).
3. Apply FP-Growth (`minSupport=0.01`, `minConfidence=0.5`).
   - Report the number of frequent itemsets and association rules.
   - Display top 20 of each.
4. Retrain FP-Growth with `minSupport=0.001`; report and display results.
5. Sort and show top 50 single-item itemsets and top 20 association rules.
6. Reflect in two sentences on the effect of varying support thresholds.
7. Implement the SON algorithm:
   - **Phase 1**: Repartition baskets into 10, mine local frequent itemsets.
   - **Phase 2**: Aggregate and validate candidates against global support (`0.001`).
   - Report candidate and final frequent itemset counts and justify potential false negatives.

## Task 3: High-Dimensional Data Analysis with K-Means & PCA
Cluster the Breast Cancer Wisconsin dataset:
1. Install PySpark and load the dataset into a Spark DataFrame with an array column `features`.
2. Run K-Means (`k=2`, `seed=1`), compute Silhouette score via `ClusteringEvaluator`.
3. Compare predicted clusters to true labels; report the maximum correctly clustered count.
4. Perform PCA reduction to 2 components; display the first 20 rows of `pcaFeatures`.
5. Rerun K-Means on PCA output; report new Silhouette score and correct classification count.
6. Comment in two sentences on clustering performance before vs. after PCA.
7. Write a concluding paragraph summarizing the impact of PCA on clustering.

## Task 4: Analyzing COVID-19 with Spark's SQL API
Using global COVID-19 time-series datasets:
1. **Setup**: Install PySpark, PyDrive, and load `time_series_covid19_confirmed_global.csv`, `...deaths_global.csv`, and `...recovered_global.csv` into Spark DataFrames.
2. **Global Totals**: Filter and sum May 1, 2021 columns to report total confirmed cases, deaths, and recoveries worldwide.
3. **Death Ratios (Mar 1 2020 vs. Mar 1 2021)**: Select regions with ≥50 confirmed cases on both dates; compute death/confirmed ratios; display top 20.
4. **Percent Increase in Deaths (Mar 1 2021 → May 1 2021)**: Filter regions with ≥50 deaths; calculate and sort percent increase; show top 20.
5. **Insights**: Conclude with key observations on global trends and regional disparities.

## Task 5: Link Analysis with NetworkX
Analyzing the Stanford Web Graph:
1. **Load Graph**: Read directed edge list from `web-Stanford.txt`; print node/edge counts and degree stats.
2. **Largest WCC**: Extract the largest weakly connected component; print its graph info.
3. **PageRank**: Compute PageRank on the WCC; list top 10 pages by rank.
4. **Graph Models**:
   - **Random Graph** (`G(n,p)` with p≈0.00008) and **Barabási–Albert** (`m ≈ edges/n`) with `n` nodes.
   - Compute and display graph stats and top 10 PageRank nodes for each.
5. **Similarity**: Compare sorted PageRank vectors (original vs. models) using cosine similarity; discuss which model better approximates the web.
6. **Conclusions**: Summarize findings on model fidelity and network structure.

## Task 6: Exploring Big Data with Embedding Techniques
Node2Vec embeddings on network data:
1. **Environment**: Install `csrgraph` and `nodevectors`; import NetworkX and Node2VecNew wrapper.
2. **Karate Club Example**: Generate 2D embeddings for Zachary’s Karate Club; visualize separation of Mr. Hi vs. Officer clusters; tune `p` & `q`.
3. **Barbell Graph**: Create two 1000-node cliques connected by no bridge; compute 10‑dim embeddings; report graph info.
4. **Similarity Function**: Write `compute_cosine_similarities(n)` to list cosine similarities of node `n` against all nodes; test on `n=5`.
5. **Neighbor Counts**:
   - **Exact 1000 Neighbors** with similarity >0.8; report count.
   - **Barbell with Path** (two 1000-node cliques + path length 51); compute embeddings; count neighbors >0.8 and <100.
6. **Hyperparameter Effects**: Analyze how `p`, `q`, and embedding dimension `d` influence local vs. global structure and similarity metrics.
7. **Conclusions**: Summarize how embedding choices impact representation quality and downstream analyses.
