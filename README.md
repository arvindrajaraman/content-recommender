# content-recommender
Content-based recommendation engine for candidates to find similar job postings to an existing job.

This recommendation engine uses Pandas, NumPy, and Scikit-Learn. The algorithms used are:
- `CountVectorizer` - to count the occurrences of job keywords in each job's "soup description"
- `CosineSimilarity` - to determine the pairwise similarity between two jobs
