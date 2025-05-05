# Talowiz_Internship_Work
 
This repository contains the work I completed during my internship at Talowiz in the months of March & April 2025.

## Contents

### Initial_Testing

1. [JD_PDFs](Initial_Testing/JD_PDFs): PDFs of different job roles given directly by the companies.
2. [jd.json](Initial_Testing/jd.json): The ideal information JSONs of all 5 test samples
3. [script.ipynb](Initial_Testing/script.ipynb): Contains script for text extraction for job description PDFs, prompt creation & example addition & inference function of Llama 3.1 8B, Gemini 1.5 Flash & CHATGPT 4-o-mini

### sim_search

 1. [JD_data_1.csv](sim_search/JD_data_1.csv) = CSV formatted dataset shared by Kartik Pujari
 2. [data_1_summaries.csv](sim_search/data_1_summaries.csv) = Contains 50 word summaries of job descriptions given in JD_Data_1.csv
 3. [JD_data_2.csv](sim_search/JD_data_2.csv) = CSV formatted dataset shared by Kartik Pujari
 4. [script.ipynb](sim_search/script.ipynb) = Contains python scripts to generate Job Description summaries from Gemini 1.5 Flash, pipeline for similarity matching using BERT embeddings on Job Titles or Job Description Summaries
5. [SimSearch.py](sim_search/SimSearch.py) = Contains 3 APIs created using the Flask Framework:

   - **/generate_json** (POST):
     - **Input:** A JSON payload with a key `"job_description"` containing the job description text.
     - **Output:** A JSON response containing the original `"job_description"` and a `"json_output"` which is a structured JSON representation of the job description, including fields like `jobRole`, `location`, `minExperience`, `maxExperience`, `mustHaveSkills`, `goodToHaveSkills`, and `matchingJobRoles`.

   - **/nearest_examples** (POST):
     - **Input:** A JSON payload with a key `"job_description"` containing the job description text to find similar examples for.
     - **Output:** A JSON response containing the `"query_summary"` (a concise summary of the input job description) and `"nearest_job_descriptions"` (a list of up to 10 job description texts from the dataset that are semantically similar to the input).

   - **/nearest_job_titles** (POST):
     - **Input:** A JSON payload with a key `"query_jd"` containing the job description text to find similar job titles for.
     - **Output:** A JSON response containing the `"query_title"` (the extracted job title from the input job description), `"nearest_job_titles"` (a list of up to 10 similar job titles from the dataset), and `"nearest_job_titles_score"` (a corresponding list of cosine similarity scores for each of the nearest job titles).
