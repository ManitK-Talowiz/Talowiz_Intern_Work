from flask import Flask, request, jsonify
import csv
import google.generativeai as genai
from time import sleep
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from dotenv import load_dotenv
import os
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("GPT_API_KEY")

app = Flask(__name__)

genai.configure(api_key=GENAI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')
model = SentenceTransformer('all-MiniLM-L6-v2')

openai.api_key = OPENAI_API_KEY

data_1_summaries_file = "/Users/manitk/Desktop/TW_Work/sim_search/data_1_summaries.csv"
data_1_summaries = []

with open(data_1_summaries_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data_1_summaries.append(row)
print(data_1_summaries[0])

data_1_file = "/Users/manitk/Desktop/TW_Work/sim_search/JD_data_1.csv"
data_1 = []

with open(data_1_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data_1.append(row)

def generate_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"].strip()


instruction = """\n You are given a job description, and you need to create a structured JSON representation using the provided information. The fields in the JSON are defined as follows:

### **Fields & Their Definitions:**

- **jobRole** (List of Strings): The specific job title(s) mentioned in the description.  
- **location** (String): The location of the job, including city, state/region, and country.  
- **minExperience** (Integer or Null): The minimum required years of experience for the role.  
- **maxExperience** (Integer or Null): The maximum required years of experience (null if not specified).  

#### **Understanding Skills:**  
A "skill" refers to a specific capability, expertise, or knowledge area required to perform a job effectively. Skills can be technical (e.g., programming languages, tools, frameworks) or professional (e.g., communication, leadership, problem-solving).  

- **mustHaveSkills** (List of Objects): A list of essential skills required for performing the core responsibilities of the role. Each skill should specify the particular tool, technology, language, or certification mentioned in the job description, along with its importance level (HIGH, MEDIUM, LOW).  
- **goodToHaveSkills** (List of Objects): A list of additional skills that are beneficial but not mandatory. These skills can enhance performance, adaptability, or efficiency in the role. Each skill should specify its importance level (HIGH, MEDIUM, LOW).  
- **matchingJobRoles** (List of Strings): Other job roles that match this job profile.  

#### **Importance Levels of Skills:**  
Each skill must have an importance level assigned based on the job description:  

- **HIGH:** A mandatory skill crucial for the role, explicitly mentioned as a requirement.  
- **MEDIUM:** A skill that is important but not strictly mandatory. Candidates with this skill will have an advantage.  
- **LOW:** A skill that is optional or nice-to-have but not necessary for the core job responsibilities.  

Use the provided job description and the above examples (if any) to extract and correctly structure this data into the JSON format.
"""


def generate_summary(jd):
    prompt = "You are given a job description. Return the summary of the job description containing only the necessary details and key words in atmost 50 words. The job description is as follows: \n"
    prompt += jd
    response = llm.generate_content(prompt)
    answer = response.text
    sleep(3)
    return answer

# can also create the embeddings for the summaries in advance and save them to a file
def nearest_neighbour(query_summary):
    indices = [int(item[0]) for item in data_1_summaries]
    summaries = [item[1] for item in data_1_summaries]
    embeddings = model.encode(summaries)
    query_embedding = model.encode([query_summary])
    cosine_similarities = cosine_similarity(query_embedding, embeddings).flatten()

    top_100_indices = np.argsort(cosine_similarities)[-100:][::-1]
    top_100_matches = [(indices[i], summaries[i], cosine_similarities[i]) for i in top_100_indices]

    unique_matches = []
    for idx, _, _ in top_100_matches:
        if idx not in unique_matches:
            unique_matches.append(idx)
        if len(unique_matches) == 10:
            break
    
    return unique_matches # indices of nearest 10 unique neighbours

def get_job_descriptions(indices):
    job_descriptions = []
    for idx in indices:
        for item in data_1:
            if int(item[0]) - 1 == idx: # -1 because of difference in indices
                job_descriptions.append(item[6])
                break
    return job_descriptions

def generate_role(job_description):
    prompt = "You are given a job description. Return the most appropriate job title of the given job description nothing else. The job description is as follows: \n"
    prompt += job_description
    response = llm.generate_content(prompt)
    answer = response.text
    sleep(3)
    return answer

def nearest_neighbour_role(query_role):
    indices = [int(item[0]) for item in data_1]
    titles = [item[4] for item in data_1]
    embeddings = model.encode(titles)
    query_embedding = model.encode([query_role])
    cosine_similarities = cosine_similarity(query_embedding, embeddings).flatten()

    top_100_indices = np.argsort(cosine_similarities)[-100:][::-1]
    top_100_matches = [(indices[i], titles[i], cosine_similarities[i]) for i in top_100_indices]

    unique_matches = []
    title_lst = []
    for _, title, score in top_100_matches:
        if title not in title_lst:
            unique_matches.append([title,score])
            title_lst.append(title)
        if len(unique_matches) == 10:
            break

    return unique_matches 

# API 1
# Request : JD text
# Output -> JSON having details like Good to Have skills, Must have skills etc
@app.route('/generate_json', methods=['POST'])
def generate_json():
    data = request.get_json()
    job_description = data.get('job_description', "")
    # add_samples_or_not = data.get('add_samples_or_not', False) ----> To add when JSON of database is available

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    # if add_samples_or_not == True:
    #     query_summary = generate_summary(job_description)
    #     nearest_indices = nearest_neighbour(query_summary)
    #     job_descriptions_examples = get_job_descriptions(nearest_indices)

    #     examples_instruct = "Given below are some examples of job description and it's information extracted in JSON representation:"
    #     for i in range(1,5): # range of examples
    #         examples_instruct += f"\n Example {i+1}: \n Job Description: {examples_data[i]}\n JSON Output: {examples_json[i]}"
    #         examples_instruct += "\n ######################## \n"

    prompt = instruction +  f"\n Job Description: {job_description}\n JSON Output:"
    gpt_json = generate_chatgpt_response(prompt)
    response = {
        "job_description": job_description,
        "json_output": gpt_json
    }
    return jsonify(response)

# API 2
# Good JD sample to be appended to LLM for API1 processing
# currently returns the nearest 10 job descriptions because of their respective JSONs not being present
@app.route('/nearest_examples', methods=['POST'])
def nearest():
    data = request.get_json()
    query_jd = data.get('job_description', "")

    if not query_jd:
        return jsonify({"error": "Job description is required"}), 400

    query_summary = generate_summary(query_jd)
    nearest_indices = nearest_neighbour(query_summary)
    job_descriptions = get_job_descriptions(nearest_indices)
    response = {
        "query_summary": query_summary,
        "nearest_job_descriptions": job_descriptions
    }
    return jsonify(response)

# API 3
# Input : JD
# Output -> Matching job roles with score
@app.route('/nearest_job_titles', methods=['POST'])
def nearest_job_titles():
    data = request.get_json()
    query_jd = data.get('query_jd', "")

    if not query_jd:
        return jsonify({"error": "Job description is required"}), 400

    query_role = generate_role(query_jd)
    nearest_roles_and_score = nearest_neighbour_role(query_role)

    nearest_roles = []
    nearest_score = []
    for role, score in nearest_roles_and_score:
        nearest_roles.append(role)
        nearest_score.append(str(score))

    response = {
        "query_title": query_role,
        "nearest_job_titles": nearest_roles,
        "nearest_job_titles_score": nearest_score
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088)

# testing api 1 -
# curl -X POST http://localhost:8088/generate_json \     -H "Content-Type: application/json" \     -d '{"job_description": "arrying out 2D and 3D magnetic designs of high-quality accelerator magnets, covering the full range of d.c., a.c. and pulsed magnets, both permanent magnet and electromagnet, using sophisticated commercially available software. Creating detailed magnetic/mechanical/electrical procurement specifications. Supervising procurement contracts for the supply of magnets. Set-up, operate and maintain high precision magnetic measurement systems. Undertake magnetic measurement of accelerator magnets and analysis of data to assess field quality. Keeping abreast of worldwide developments in magnet technology. The major focus of this role is pulsed magnets i.e. kickers and septa."}'

# testing api 2 -
# curl -X POST http://localhost:8088/nearest_examples \     -H "Content-Type: application/json" \     -d '{"job_description": "arrying out 2D and 3D magnetic designs of high-quality accelerator magnets, covering the full range of d.c., a.c. and pulsed magnets, both permanent magnet and electromagnet, using sophisticated commercially available software. Creating detailed magnetic/mechanical/electrical procurement specifications. Supervising procurement contracts for the supply of magnets. Set-up, operate and maintain high precision magnetic measurement systems. Undertake magnetic measurement of accelerator magnets and analysis of data to assess field quality. Keeping abreast of worldwide developments in magnet technology. The major focus of this role is pulsed magnets i.e. kickers and septa."}'

# testing api 3 - 
# curl -X POST http://localhost:8088/nearest_job_titles \     -H "Content-Type: application/json" \     -d '{"query_jd": "arrying out 2D and 3D magnetic designs of high-quality accelerator magnets, covering the full range of d.c., a.c. and pulsed magnets, both permanent magnet and electromagnet, using sophisticated commercially available software. Creating detailed magnetic/mechanical/electrical procurement specifications. Supervising procurement contracts for the supply of magnets. Set-up, operate and maintain high precision magnetic measurement systems. Undertake magnetic measurement of accelerator magnets and analysis of data to assess field quality. Keeping abreast of worldwide developments in magnet technology. The major focus of this role is pulsed magnets i.e. kickers and septa."}'