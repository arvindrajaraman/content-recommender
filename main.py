import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_str(s, condense=False):
    if isinstance(s, str):
        s = re.sub('[^0-9a-zA-Z]+', ' ', s)
        if condense:
            s = ''.join(s.split())
        else:
            s = ' '.join(s.split())
        s = s.lower()
        return s
    return clean_str(str(s))


def condense_str(s):
    return clean_str(s, True)


def to_list(csl):
    if isinstance(csl, str):
        words = csl.split(',')
        return [condense_str(word) for word in words]
    elif isinstance(csl, list):
        return csl
    return []


def to_date(x):
    return pd.to_datetime(x, format='%Y-%m-%d', errors='ignore')


def rep(s, t=1):
    return (s + ' ') * t


def create_soup(x):
    classes = rep(x['title']) + rep(x['employment_terms'], 1) + rep(x['job_type'], 1) + rep(x['category']) + rep(x['candidate_level'])
    description = x['required_qualifications'] + ' ' + x['responsibilities'] + ' ' + ' '.join(x['soft_skills']) + ' ' + ' '.join(x['prof_skills'])
    return classes + description


df = pd.read_csv('staff.am_data_2020.csv')

# Clean strings
df['title'] = df['title'].apply(condense_str)
df['employment_terms'] = df['employment_terms'].apply(condense_str)
df['job_type'] = df['job_type'].apply(condense_str)
df['category'] = df['category'].apply(condense_str)
df['required_qualifications'] = df['required_qualifications'].apply(clean_str)
df['responsibilities'] = df['responsibilities'].apply(clean_str)
df['candidate_level'] = df['candidate_level'].apply(condense_str)

# Convert comma-separated lists into Python lists
df['soft_skills'] = df['soft_skills'].apply(to_list)
df['prof_skills'] = df['prof_skills'].apply(to_list)

# Convert to date
df['deadline'] = df['deadline'].apply(to_date)

df['soup'] = df.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_mat = count.fit_transform(df['soup'])

cos_sim = cosine_similarity(count_mat, count_mat)
df = df.reset_index()
indices = pd.Series(df.index, index=df['id'])

idx = indices[id]
sim_scores = list(enumerate(cos_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:11]
job_indices = [i[0] for i in sim_scores]
