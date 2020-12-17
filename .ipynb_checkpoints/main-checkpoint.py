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


def rep(s, t=5):
    return (s + ' ') * t


def create_soup(x):
    classes = x['title'] + rep(x['employment_terms']) + rep(x['job_type']) + rep(x['category']) + rep(x['candidate_level'])
    description = x['required_qualifications'] + ' ' + x['responsibilities'] + ' ' + ' '.join(x['soft_skills']) + ' ' + ' '.join(x['prof_skills'])
    return classes + description


df = pd.read_csv('staff.am_data_2020.csv')

# Add test job
test_job = {
    'id': 'test_job',
    'title': 'Marketing Specialist',
    'employment_terms': 'Permanent',
    'job_type': 'Full time',
    'deadline': '2020-12-31',
    'category': 'Marketing/Advertising',
    'required_qualifications': """Education background does not matter; we are looking for talent, experience, passion and creativity.
        At least 1 year of professional full-time experience in social media marketing is preferred.
        Experience with running social media pages such as Facebook is a must, experience with Linkedin, Insta and YouTube is a plus.
        Excellent in creative content writing, 
        Experience and knowledge of Adobe Photoshop (knowledge of other tools is a plus), for creating social media postings, based on visual templates provided by the designer.
        Strong written and verbal communication in Armenian and English languages, Russian is desirable.
        Positive attitude, detail and customer oriented with good multitasking and organisational ability.""",
    'responsibilities': """Develop original and exciting SMM campaigns and content on a daily basis (e.g. social media posts, website content, etc).
        Coordinate with marketing and design teams to generate relevant marketing content, 
        Manage staff.am's & HireBee's social media presence on Facebook, Linkedin, Telegram, Instagram and YouTube.
        Prepare successful email marketing campaigns with well-structured content.
        Maintain appropriate tone of voice through social media and other digital channels.
        Suggest and implement other marketing activities to boost awareness and increase website traffic and app installs.
        Complete other tasks related to Marketing as required.""",
    'soft_skills': 'Written communication skills,Positive attitude,Time management,Team player',
    'prof_skills': 'Adobe Photoshop,SMM,Email Marketing,Content marketing',
    'salary': 'NaN',
    'candidate_level': 'Mid level'
}
df = df.append(test_job, ignore_index=True)

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
df['deadline'] = df['deadline'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='ignore'))

# Create soup
df['soup'] = df.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_mat = count.fit_transform(df['soup'])

cos_sim = cosine_similarity(count_mat, count_mat)
df = df.reset_index()
indices = pd.Series(df.index, index=df['id'])

def get_recommendations(id, cos_sim=cos_sim):
    idx = indices[id]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    job_indices = [i[0] for i in sim_scores]
    return df.iloc[job_indices]

get_recommendations('test_job', cos_sim)