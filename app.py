import streamlit as st
import pandas as pd
import numpy as np
import re
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pyvis.network import Network
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 15px;
        margin-bottom: 15px;
    }

    .skill-selected {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Course Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find the most suitable courses based on your skills, interests, and preferred level.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("Coursera.csv")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Possible column aliases
    aliases = {

        "course_name": [
            "course_name",
            "course",
            "name",
            "course_title"
        ],

        "difficulty_level": [
            "difficulty_level",
            "level",
            "difficulty"
        ],

        "course_rating": [
            "course_rating",
            "rating",
            "course_rate"
        ],

        "course_description": [
            "course_description",
            "description",
            "course_desc"
        ],

        "skills": [
            "skills",
            "skill"
        ],

        "category": [
            "category",
            "categories",
            "subject"
        ]
    }

    # Rename matching columns
    rename_map = {}

    for standard_name, possible_names in aliases.items():

        for possible_name in possible_names:

            if possible_name in df.columns:

                rename_map[possible_name] = standard_name
                break

    df = df.rename(columns=rename_map)

    # Create missing columns
    required_columns = [
        "course_name",
        "difficulty_level",
        "course_rating",
        "course_description",
        "skills",
        "category"
    ]

    for col in required_columns:

        if col not in df.columns:
            df[col] = ""

    # Convert columns to string
    for col in [
        "course_name",
        "difficulty_level",
        "course_description",
        "skills",
        "category"
    ]:

        df[col] = df[col].fillna("").astype(str)

    # Convert rating to numeric
    df["course_rating"] = pd.to_numeric(
        df["course_rating"],
        errors="coerce"
    )

    df["course_rating"] = df["course_rating"].fillna(0)

    return df


df = load_dataset()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# CATEGORY INFERENCE
# ============================================================

def infer_category(row):

    # If original category exists, use it
    original_category = str(row["category"]).strip()

    if original_category:
        return original_category

    text = " ".join([
        str(row["course_name"]),
        str(row["skills"]),
        str(row["course_description"])
    ]).lower()

    category_keywords = {

        "Artificial Intelligence": [
            "artificial intelligence",
            " ai ",
            "intelligent systems",
            "generative ai",
            "llm",
            "large language model"
        ],

        "Machine Learning": [
            "machine learning",
            "supervised learning",
            "unsupervised learning",
            "classification",
            "regression",
            "clustering",
            "xgboost",
            "random forest"
        ],

        "Data Science": [
            "data science",
            "data scientist",
            "data mining",
            "data analysis",
            "pandas",
            "numpy"
        ],

        "Deep Learning": [
            "deep learning",
            "neural network",
            "neural networks",
            "tensorflow",
            "pytorch",
            "cnn",
            "rnn"
        ],

        "Programming": [
            "python programming",
            "programming",
            "java programming",
            "c programming",
            "javascript",
            "coding"
        ],

        "Computer Science": [
            "computer science",
            "algorithms",
            "data structures",
            "operating systems",
            "computer architecture"
        ],

        "Data Analysis": [
            "data analysis",
            "excel",
            "tableau",
            "power bi",
            "statistics",
            "visualization"
        ],

        "Cybersecurity": [
            "cybersecurity",
            "cyber security",
            "network security",
            "information security",
            "ethical hacking"
        ],

        "Cloud Computing": [
            "cloud computing",
            "aws",
            "azure",
            "google cloud",
            "cloud infrastructure"
        ],

        "Business": [
            "business",
            "marketing",
            "management",
            "finance",
            "accounting",
            "leadership"
        ],

        "Computer Vision": [
            "computer vision",
            "image processing",
            "object detection",
            "image classification"
        ],

        "Natural Language Processing": [
            "natural language processing",
            "nlp",
            "text mining",
            "sentiment analysis"
        ]
    }

    # Longer / more specific categories first
    for category, keywords in category_keywords.items():

        for keyword in keywords:

            if keyword.strip() in text:
                return category

    return "General"


df["inferred_category"] = df.apply(
    infer_category,
    axis=1
)


# ============================================================
# BUILD TF-IDF RECOMMENDER
# ============================================================

@st.cache_resource
def build_recommender(data):

    course_profiles = []

    for _, row in data.iterrows():

        profile = " ".join([

            clean_text(row["course_name"]),
            clean_text(row["course_name"]),

            clean_text(row["skills"]),
            clean_text(row["skills"]),

            clean_text(row["course_description"]),

            clean_text(row["difficulty_level"]),

            clean_text(row["inferred_category"])
        ])

        course_profiles.append(profile)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(course_profiles)

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = build_recommender(df)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_courses(
    skills,
    interests,
    level,
    top_n=5
):

    # Build user profile
    user_profile = " ".join([

        clean_text(skills),
        clean_text(skills),

        clean_text(interests),

        clean_text(level)
    ])

    # Transform user profile
    user_vector = vectorizer.transform(
        [user_profile]
    )

    # Cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # Level boost
    if level:

        level_clean = clean_text(level)

        course_levels = (
            df["difficulty_level"]
            .apply(clean_text)
        )

        level_match = (
            course_levels
            .str.contains(
                level_clean,
                regex=False
            )
            .astype(float)
            .values
        )

        # 90% content similarity
        # 10% preferred level
        similarity_scores = (
            similarity_scores * 0.90
            + level_match * 0.10
        )

    # Create result dataframe
    results = df.copy()

    results["similarity_score"] = similarity_scores

    # Sort
    results = results.sort_values(
        by="similarity_score",
        ascending=False
    )

    # Remove duplicate course names
    results = results.drop_duplicates(
        subset=["course_name"]
    )

    return results.head(top_n)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_skills" not in st.session_state:

    st.session_state.selected_skills = []


if "recommendations" not in st.session_state:

    st.session_state.recommendations = None


# ============================================================
# TABS
# ============================================================

recommendation_tab, dataset_tab = st.tabs(
    [
        "🎯 Course Recommendation",
        "📊 Dataset Information"
    ]
)


# ============================================================
# RECOMMENDATION TAB
# ============================================================

with recommendation_tab:

    st.markdown(
        '<div class="section-title">🧠 Tell us what you want to learn</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Select your skills and describe your interests to get personalized course recommendations.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # SKILLS
    # ========================================================

    st.subheader("💡 Select Your Skills")

    st.write(
        "Click on the skills you already know or want to use."
    )


    skill_suggestions = [

        "Python",
        "SQL",
        "Machine Learning",
        "Pandas",
        "NumPy",
        "Deep Learning",
        "NLP",
        "TensorFlow",
        "PyTorch",
        "Data Analysis",
        "Data Science",
        "Computer Vision"

    ]


    # Create 3 columns
    skill_columns = st.columns(3)


    for index, skill in enumerate(skill_suggestions):

        column = skill_columns[index % 3]

        is_selected = (
            skill in st.session_state.selected_skills
        )

        button_text = (
            f"✓ {skill}"
            if is_selected
            else skill
        )

        with column:

            if st.button(
                button_text,
                key=f"skill_{index}",
                use_container_width=True
            ):

                if skill in st.session_state.selected_skills:

                    st.session_state.selected_skills.remove(skill)

                else:

                    st.session_state.selected_skills.append(skill)

                st.rerun()


    # ========================================================
    # SELECTED SKILLS
    # ========================================================

    if st.session_state.selected_skills:

        st.markdown("### Selected Skills")

        selected_text = ", ".join(
            st.session_state.selected_skills
        )

        st.text_area(
            "Selected Skills",
            value=selected_text,
            height=70,
            disabled=True,
            label_visibility="collapsed"
        )

        if st.button(
            "🗑️ Clear Selected Skills",
            use_container_width=False
        ):

            st.session_state.selected_skills = []

            st.rerun()


    # ========================================================
    # INTERESTS
    # ========================================================

    st.markdown("---")

    st.subheader("❤️ Your Interests")

    interests = st.text_area(
        "Your Interests",
        placeholder=(
            "Example: I am interested in Artificial Intelligence, "
            "Machine Learning, NLP, and building AI applications."
        ),
        height=200,
        label_visibility="collapsed"
    )


    # ========================================================
    # LEVEL
    # ========================================================

    st.subheader("📈 Preferred Course Level")

    user_level = st.selectbox(

        "Choose your preferred course level",

        options=[
            "Beginner",
            "Intermediate",
            "Advanced"
        ],

        index=None,

        placeholder="Select your preferred level...",

        label_visibility="collapsed"
    )


    # ========================================================
    # RECOMMEND BUTTON
    # ========================================================

    st.markdown("---")

    recommend_button = st.button(
        "🚀 Get Course Recommendations",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN RECOMMENDATION
    # ========================================================

    if recommend_button:

        user_skills = ", ".join(
            st.session_state.selected_skills
        )

        # Validate input
        if not user_skills and not interests.strip():

            st.warning(
                "Please select at least one skill or enter your interests."
            )

        else:

            level_value = (
                user_level
                if user_level
                else ""
            )

            results = recommend_courses(

                skills=user_skills,

                interests=interests,

                level=level_value,

                top_n=5
            )

            st.session_state.recommendations = results


    # ========================================================
    # DISPLAY RECOMMENDATIONS
    # ========================================================

    if st.session_state.recommendations is not None:

        results = st.session_state.recommendations

        st.markdown("---")

        st.header("🎓 Recommended Courses")

        st.write(
            "Based on your selected skills, interests, and preferred level:"
        )


        # ====================================================
        # COURSE CARDS
        # ====================================================

        for index, (_, course) in enumerate(
            results.iterrows(),
            start=1
        ):

            course_name = str(
                course["course_name"]
            ).strip()

            category = str(
                course["inferred_category"]
            ).strip()

            level = str(
                course["difficulty_level"]
            ).strip()

            rating = float(
                course["course_rating"]
            )

            similarity = (
                float(course["similarity_score"])
                * 100
            )


            # Native Streamlit container
            # instead of HTML to prevent raw HTML display
            with st.container(border=True):

                st.subheader(
                    f"{index}. {course_name}"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Category",
                        category
                    )

                with col2:

                    st.metric(
                        "Level",
                        level if level else "Not specified"
                    )

                with col3:

                    st.metric(
                        "Rating",
                        f"{rating:.1f}/5"
                    )

                with col4:

                    st.metric(
                        "Similarity",
                        f"{similarity:.2f}%"
                    )


                description = str(
                    course["course_description"]
                ).strip()


                if description:

                    with st.expander(
                        "📖 Course Description"
                    ):

                        st.write(description)


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.markdown("---")

        st.header("📋 Recommendation Results")

        display_results = results[
            [
                "course_name",
                "inferred_category",
                "difficulty_level",
                "course_rating",
                "similarity_score"
            ]
        ].copy()


        display_results.columns = [

            "Course",

            "Category",

            "Level",

            "Rating",

            "Similarity Score"

        ]


        display_results["Similarity Score"] = (

            display_results["Similarity Score"] * 100

        ).round(2).astype(str) + "%"


        display_results["Rating"] = (

            display_results["Rating"]
            .round(1)
        )


        st.dataframe(

            display_results,

            use_container_width=True,

            hide_index=True
        )


        # ====================================================
        # KNOWLEDGE GRAPH
        # ====================================================

        st.markdown("---")

        st.header("🕸️ Knowledge Graph")

        st.write(
            "The graph connects the user with their skills, "
            "related topics, and recommended courses."
        )


        graph = nx.DiGraph()


        # User node
        graph.add_node(
            "User",
            label="User",
            title="User",
            group="user"
        )


        # ====================================================
        # SKILL → TOPIC MAPPING
        # ====================================================

        skill_topic_mapping = {

            "python": [
                "Programming",
                "Data Science",
                "Machine Learning"
            ],

            "sql": [
                "Databases",
                "Data Analysis"
            ],

            "machine learning": [
                "Artificial Intelligence",
                "Data Science"
            ],

            "pandas": [
                "Data Science",
                "Data Analysis"
            ],

            "numpy": [
                "Data Science",
                "Data Analysis"
            ],

            "deep learning": [
                "Artificial Intelligence",
                "Neural Networks"
            ],

            "nlp": [
                "Artificial Intelligence",
                "Natural Language Processing"
            ],

            "tensorflow": [
                "Deep Learning",
                "Artificial Intelligence"
            ],

            "pytorch": [
                "Deep Learning",
                "Artificial Intelligence"
            ],

            "data analysis": [
                "Data Analysis",
                "Data Science"
            ],

            "data science": [
                "Data Science",
                "Machine Learning"
            ],

            "computer vision": [
                "Artificial Intelligence",
                "Computer Vision"
            ]
        }


        # ====================================================
        # USER SKILLS
        # ====================================================

        for skill in st.session_state.selected_skills:

            skill_node = f"Skill: {skill}"

            graph.add_node(
                skill_node,
                label=skill,
                title=f"Skill: {skill}",
                group="skill"
            )

            graph.add_edge(
                "User",
                skill_node,
                label="HAS_SKILL"
            )


            # Topics
            topics = skill_topic_mapping.get(
                skill.lower(),
                ["Technology"]
            )


            for topic in topics:

                topic_node = f"Topic: {topic}"

                graph.add_node(
                    topic_node,
                    label=topic,
                    title=f"Topic: {topic}",
                    group="topic"
                )

                graph.add_edge(
                    skill_node,
                    topic_node,
                    label="RELATED_TO"
                )


        # ====================================================
        # INTEREST TOPICS
        # ====================================================

        if interests.strip():

            interest_text = interests.lower()

            interest_keywords = {

                "Artificial Intelligence": [
                    "artificial intelligence",
                    " ai ",
                    "generative ai"
                ],

                "Machine Learning": [
                    "machine learning",
                    "ml"
                ],

                "Data Science": [
                    "data science"
                ],

                "Deep Learning": [
                    "deep learning"
                ],

                "NLP": [
                    "nlp",
                    "natural language processing"
                ],

                "Computer Vision": [
                    "computer vision"
                ],

                "Programming": [
                    "programming",
                    "python",
                    "coding"
                ],

                "Data Analysis": [
                    "data analysis",
                    "analytics"
                ]

            }


            for topic, keywords in interest_keywords.items():

                if any(
                    keyword in interest_text
                    for keyword in keywords
                ):

                    topic_node = f"Interest: {topic}"

                    graph.add_node(
                        topic_node,
                        label=topic,
                        title=f"Interest: {topic}",
                        group="topic"
                    )

                    graph.add_edge(
                        "User",
                        topic_node,
                        label="INTERESTED_IN"
                    )


        # ====================================================
        # COURSE NODES
        # ====================================================

        for index, (_, course) in enumerate(
            results.iterrows()
        ):

            course_name = str(
                course["course_name"]
            ).strip()

            course_node = (
                f"Course {index + 1}: {course_name}"
            )


            graph.add_node(

                course_node,

                label=course_name,

                title=(
                    f"{course_name}<br>"
                    f"Category: {course['inferred_category']}<br>"
                    f"Level: {course['difficulty_level']}<br>"
                    f"Rating: {course['course_rating']:.1f}<br>"
                    f"Similarity: "
                    f"{course['similarity_score'] * 100:.2f}%"
                ),

                group="course"
            )


            # User recommends course
            graph.add_edge(

                "User",

                course_node,

                label="RECOMMENDS"
            )


            # Connect selected skills to course
            for skill in st.session_state.selected_skills:

                skill_node = f"Skill: {skill}"

                if skill_node in graph.nodes:

                    graph.add_edge(

                        skill_node,

                        course_node,

                        label="TEACHES"
                    )


            # Connect category/topic to course
            category = str(
                course["inferred_category"]
            ).strip()

            if category:

                topic_node = f"Topic: {category}"

                if topic_node in graph.nodes:

                    graph.add_edge(

                        topic_node,

                        course_node,

                        label="RECOMMENDS"
                    )


        # ====================================================
        # PYVIS GRAPH
        # ====================================================

        net = Network(

            height="700px",

            width="100%",

            directed=True,

            bgcolor="#ffffff",

            font_color="#222222"

        )


        net.from_nx(graph)


        net.set_options(
            """
            {
              "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true,
                "dragView": true
              },

              "physics": {
                "enabled": true,
                "stabilization": {
                  "iterations": 150
                },

                "barnesHut": {
                  "gravitationalConstant": -3000,
                  "centralGravity": 0.2,
                  "springLength": 180,
                  "springConstant": 0.04,
                  "damping": 0.09
                }
              },

              "edges": {
                "arrows": {
                  "to": {
                    "enabled": true
                  }
                },

                "smooth": {
                  "enabled": true
                }
              },

              "nodes": {
                "font": {
                  "size": 16
                }
              }
            }
            """
        )


        graph_file = "knowledge_graph.html"

        net.save_graph(graph_file)


        with open(
            graph_file,
            "r",
            encoding="utf-8"
        ) as file:

            graph_html = file.read()


        components.html(

            graph_html,

            height=720,

            scrolling=True

        )


# ============================================================
# DATASET INFORMATION TAB
# ============================================================

with dataset_tab:

    st.header("📊 Coursera Dataset Information")

    st.write(
        "This section provides an overview of the dataset used "
        "by the recommendation system."
    )


    # ========================================================
    # DATASET METRICS
    # ========================================================

    total_courses = len(df)

    total_columns = len(df.columns)

    average_rating = df[
        "course_rating"
    ].mean()

    number_of_levels = (
        df["difficulty_level"]
        .replace("", np.nan)
        .nunique()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Courses",
            f"{total_courses:,}"
        )


    with col2:

        st.metric(
            "Dataset Columns",
            total_columns
        )


    with col3:

        st.metric(
            "Average Rating",
            f"{average_rating:.2f}"
        )


    with col4:

        st.metric(
            "Course Levels",
            number_of_levels
        )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.markdown("---")

    st.subheader("👀 Dataset Preview")

    st.dataframe(

        df.head(10),

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader("🧾 Dataset Columns")


    column_information = pd.DataFrame({

        "Column": [

            "Course Name",

            "University",

            "Difficulty Level",

            "Course Rating",

            "Course URL",

            "Course Description",

            "Skills",

            "Category"

        ],

        "Description": [

            "Name of the course.",

            "University or organization providing the course.",

            "Difficulty level of the course.",

            "Rating given to the course.",

            "URL of the course.",

            "Description of the course content.",

            "Skills associated with the course.",

            "Category inferred from the course content when not provided by the dataset."

        ]

    })


    st.dataframe(

        column_information,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # LEVEL DISTRIBUTION
    # ========================================================

    st.markdown("---")

    st.subheader("📈 Difficulty Level Distribution")


    level_counts = (

        df["difficulty_level"]

        .replace("", "Not Specified")

        .value_counts()

    )


    st.bar_chart(level_counts)


    # ========================================================
    # RATING DISTRIBUTION
    # ========================================================

    st.subheader("⭐ Rating Distribution")


    rating_data = (

        df["course_rating"]

        .round(1)

        .value_counts()

        .sort_index()

    )


    st.bar_chart(rating_data)


    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    st.subheader("📚 Course Category Distribution")


    category_counts = (

        df["inferred_category"]

        .value_counts()

        .head(15)

    )


    st.bar_chart(category_counts)


    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader("🤖 Recommendation Method")


    st.write(
        """
        **Content-Based Recommendation**

        The system uses TF-IDF (Term Frequency-Inverse Document Frequency)
        to convert course information into numerical vectors.

        The user profile is created from:

        - Selected skills
        - User interests
        - Preferred difficulty level

        Cosine Similarity is then used to measure the similarity between
        the user profile and every course in the dataset.

        The system returns the Top 5 courses with the highest similarity
        scores.

        **Knowledge Graph**

        The Knowledge Graph represents:

        User → Skills → Topics → Courses

        This provides a visual representation of how the user's skills
        and interests are connected to the recommended courses.
        """
    )