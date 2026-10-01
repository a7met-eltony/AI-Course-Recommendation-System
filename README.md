# 🎓 AI Course Recommendation System

An interactive Streamlit application that recommends online courses based on a user's **skills, interests, and learning level**.

## Features

- Content-Based Recommendation
- TF-IDF vectorization
- Cosine Similarity scoring
- Top 3–5 course recommendations
- Similarity scores for every recommendation
- Interactive Knowledge Graph using NetworkX + PyVis
- User → Skills → Topics → Courses relationships
- Course metadata: category, level, rating, description

## Dataset

The project uses the **Coursera Courses Dataset 2021**. The expected CSV filename is:

```text
Coursera.csv
```

Place it in the same directory as `app.py`.

## Project Structure

```text
AI-Course-Recommendation-System/
├── app.py
├── Coursera.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## GitHub

```bash
git init
git add .
git commit -m "Build AI course recommendation Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Course-Recommendation-System.git
git push -u origin main
```

## Deploy with Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the repository and branch `main`.
5. Set the main file to `app.py`.
6. Deploy.

`requirements.txt` is included so the deployment environment can install the required packages.
