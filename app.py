import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import altair as alt

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Student Pass/Fail Predictor")

# --- Constants and Configurations ---
PASS_THRESHOLD = 40 # Students with marks >= 40 are considered 'Pass' for each subject

# --- Utility Functions ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads the student data, calculates average marks, and creates a 'pass_fail' target.
    """
    try:
        df = pd.read_csv(file_path)

        # Define subject columns for calculations
        subject_cols = ['maths_marks', 'science_marks', 'english_marks',
                        'social_studies_marks', 'language_marks']

        for col in subject_cols:
            if col not in df.columns:
                st.error(f"Missing expected column: {col}. Please check your CSV file.")
                return None, None
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=subject_cols, inplace=True)
        df['average_marks'] = df[subject_cols].mean(axis=1)
        
        # New logic: Pass/Fail based on passing ALL subjects
        for col in subject_cols:
            df[f'{col}_status'] = (df[col] >= PASS_THRESHOLD).astype(int)
        
        df['overall_pass_fail'] = df[[f'{col}_status' for col in subject_cols]].min(axis=1)

        return df, subject_cols
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None

@st.cache_resource
def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    return model

# --- Streamlit Application Layout ---

st.title("🎓 Student Pass/Fail Predictor")
st.markdown("---")

# Initialize session state for navigation if not already present
if 'page' not in st.session_state:
    st.session_state.page = '1. Dataset Overview'

# --- Sidebar for Navigation and Information ---
with st.sidebar:
    st.header("App Navigation")
    
    # Create navigation buttons in the sidebar
    if st.button("1. Dataset Overview", use_container_width=True):
        st.session_state.page = '1. Dataset Overview'
    if st.button("2. Model Performance", use_container_width=True):
        st.session_state.page = '2. Model Performance'
    if st.button("3. Pass/Fail Distribution", use_container_width=True):
        st.session_state.page = '3. Pass/Fail Distribution'
    if st.button("4. Subject Analysis", use_container_width=True):
        st.session_state.page = '4. Subject Analysis'
    if st.button("5. New Student Predictor", use_container_width=True):
        st.session_state.page = '5. New Student Predictor'
    if st.button("6. All Predictions", use_container_width=True):
        st.session_state.page = '6. All Predictions'
        
    st.markdown("---") 

# --- Main Content Sections ---

# Load and preprocess data only once
df, subject_cols = load_and_preprocess_data('student_mat.csv')

if df is not None and subject_cols is not None:
    # Use the new 'overall_pass_fail' for the model's target variable
    X = df[subject_cols]
    y = df['overall_pass_fail']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # --- Conditional rendering of sections based on button clicks ---
    if st.session_state.page == '1. Dataset Overview':
        st.header("1. Dataset Overview")
        st.write("Here's a glimpse of the processed dataset:")
        st.dataframe(df.head())

        st.write("Dataset Statistics:")
        st.dataframe(df.describe())

    elif st.session_state.page == '2. Model Performance':
        st.header("2. Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Model Accuracy", value=f"{accuracy_score(y_test, y_pred):.2f}")
        with col2:
            st.write("#### Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.round(2))
            st.info("0: Fail, 1: Pass")

    elif st.session_state.page == '3. Pass/Fail Distribution':
        st.header("3. Pass/Fail Distribution")
        st.write("Distribution of 'Pass' and 'Fail' students in the dataset:")
        
        pass_fail_counts = df['overall_pass_fail'].value_counts().rename(index={0: 'Fail', 1: 'Pass'}).reset_index()
        pass_fail_counts.columns = ['Status', 'Count']
        
        color_scale = alt.Scale(domain=['Pass', 'Fail'], range=['#4CAF50', '#F44336'])
        
        chart = alt.Chart(pass_fail_counts).mark_bar().encode(
            x=alt.X('Status:N', axis=None),
            y=alt.Y('Count:Q'),
            color=alt.Color('Status:N', scale=color_scale, legend=None),
            tooltip=['Status', 'Count']
        ).properties(
            title='Student Status Distribution'
        )
        st.altair_chart(chart, use_container_width=True)
        st.info(f"Based on a passing score of {PASS_THRESHOLD} for all subjects.")

    elif st.session_state.page == '4. Subject Analysis':
        st.header("4. Subject Performance Analysis")
        st.write("Average marks for each subject across all students:")
        subject_average_marks = df[subject_cols].mean().rename(index={
            'maths_marks': 'Maths',
            'science_marks': 'Science',
            'english_marks': 'English',
            'social_studies_marks': 'Social Studies',
            'language_marks': 'Language'
        })
        
        pie_data = pd.DataFrame({
            'Subject': subject_average_marks.index,
            'Average Mark': subject_average_marks.values
        })

        col_graph, col_pie = st.columns(2)
        
        color_scheme = 'plasma'

        with col_graph:
            st.subheader("Average Marks per Subject (Bar Chart)")
            bar_chart = alt.Chart(pie_data).mark_bar().encode(
                x=alt.X('Subject', sort=None, title=None),
                y=alt.Y('Average Mark', title='Average Mark'),
                color=alt.Color('Subject', legend=None, scale=alt.Scale(scheme=color_scheme)),
                tooltip=['Subject', 'Average Mark']
            ).properties(
                title='Average Marks per Subject'
            )
            st.altair_chart(bar_chart, use_container_width=True)

        with col_pie:
            st.subheader("Average Marks per Subject (Pie Chart)")
            pie_chart = alt.Chart(pie_data).mark_arc(outerRadius=120).encode(
                theta=alt.Theta("Average Mark", stack=True),
                color=alt.Color("Subject", scale=alt.Scale(scheme=color_scheme)),
                order=alt.Order("Average Mark", sort="descending"),
                tooltip=["Subject", "Average Mark"]
            ).properties(
                title='Proportion of Marks by Subject'
            )
            text = pie_chart.mark_text(radius=140).encode(
                text=alt.Text("Subject"),
                order=alt.Order("Average Mark", sort="descending"),
                color=alt.value("black")
            )
            final_pie_chart = pie_chart + text
            st.altair_chart(final_pie_chart, use_container_width=True)

    elif st.session_state.page == '5. New Student Predictor':
        st.header("5. Predict for a New Student")
        st.write("Enter the marks for a new student to predict if they will pass or fail.")

        with st.form("new_student_form"):
            st.subheader("Enter Marks (0-100)")
            maths = st.slider("Maths Marks", 0, 100, 70)
            science = st.slider("Science Marks", 0, 100, 65)
            english = st.slider("English Marks", 0, 100, 75)
            social_studies = st.slider("Social Studies Marks", 0, 100, 50)
            language = st.slider("Language Marks", 0, 100, 80)

            submitted = st.form_submit_button("Predict Pass/Fail")

            if submitted:
                # Individual subject pass/fail calculation
                individual_results = {
                    'Maths': maths >= PASS_THRESHOLD,
                    'Science': science >= PASS_THRESHOLD,
                    'English': english >= PASS_THRESHOLD,
                    'Social Studies': social_studies >= PASS_THRESHOLD,
                    'Language': language >= PASS_THRESHOLD
                }
                
                # Check overall status
                overall_pass = all(individual_results.values())
                
                st.subheader("Subject-wise Performance:")
                
                # Display individual results using metrics for a clean look
                cols = st.columns(5)
                subjects = ['Maths', 'Science', 'English', 'Social Studies', 'Language']
                marks = [maths, science, english, social_studies, language]
                
                for i, col in enumerate(cols):
                    subject_name = subjects[i]
                    mark = marks[i]
                    status = "✅ Pass" if individual_results[subject_name] else "❌ Fail"
                    color = "green" if individual_results[subject_name] else "red"
                    col.markdown(f"**{subject_name}:**")
                    col.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)
                    col.metric(label="Score", value=f"{mark}")
                
                st.markdown("---")
                
                st.subheader("Overall Prediction Result:")
                if overall_pass:
                    st.success(f"🥳 This student is predicted to **PASS!**")
                else:
                    st.error(f"😔 This student is predicted to **FAIL.** They need to pass all subjects to pass overall.")

                # The model's prediction is still available for a more advanced prediction
                new_student_data = pd.DataFrame([[maths, science, english, social_studies, language]],
                                                columns=subject_cols)
                
                prediction_proba = model.predict_proba(new_student_data)[0]
                
                st.info(f"The machine learning model's prediction for an overall pass is with {prediction_proba[1]*100:.2f}% confidence.")


    elif st.session_state.page == '6. All Predictions':
        st.header("6. Predictions for All Students in Dataset")
        st.write("Here are the pass/fail predictions for each student present in the `student_mat.csv` dataset, based on the trained model and the rule of passing all subjects.")

        all_student_predictions = model.predict(X)
        all_student_prediction_proba = model.predict_proba(X)

        results_df = df[['student_name', 'average_marks'] + subject_cols].copy()
        
        # Add individual subject statuses
        for col in subject_cols:
            results_df[f'{col.replace("_marks", "")} Status'] = np.where(results_df[col] >= PASS_THRESHOLD, 'Pass', 'Fail')
            
        results_df['Predicted Status'] = np.where(all_student_predictions == 1, 'Pass', 'Fail')
        results_df['Confidence (Pass)'] = all_student_prediction_proba[:, 1]
        results_df['Confidence (Fail)'] = all_student_prediction_proba[:, 0]

        st.dataframe(results_df.round(2))
        
else:
    st.error("Could not load or process the dataset. Please ensure 'student_mat.csv' is in the same directory and has the correct columns.")