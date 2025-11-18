**🎓 Student Dropout Probability Predictor**

Follow the steps in this guide to see how I analyzed the data, trained 10 different models, and built a live web app to predict the results in real-time.


**🚀 Key Components**
Before you start, here's what you'll find inside this repository:

A Live Web App: Check app/app.py. This is a Streamlit app you can run locally to get real time predictions.

Probability Scores: The app doesn't just give a "yes" or "no." You'll see the exact probability for all three outcomes.

Full Data Analysis: The 1_EDA_and_Preprocessing.ipynb notebook is your starting point. It shows you exactly how I explored and understood the data.

Model Showdown: In 2_Model_Training.ipynb, you'll see how 10 different ML models were trained and compared to find the best one.



**📁 Understand the Folder Structure:**
Get familiar with how the files are organized. This is a standard layout for an ML project:
<img width="692" height="358" alt="image" src="https://github.com/user-attachments/assets/10e444f8-0b70-497d-84e6-a54287b0a684" />

**Methodology: A Step by Step Guide:**
To understand the project, follow the same steps we did. I've split the work into two main notebooks.

**Step 1:** Understand the Data (EDA)
File: notebooks/1_EDA_and_Preprocessing.ipynb

Your first step is to open notebooks/1_EDA_and_Preprocessing.ipynb. Here's what you need to look for in that file:

Analyze the Target: Look at the first plot for the Target variable. You'll immediately see it's imbalanced (way more "Graduate" students than "Dropout"). This is the most important problem you need to be aware of.

Find Strong Predictors: Check the plots for features like 'Tuition fees up to date' and 'Debtor'. You'll see how these financial factors are strongly correlated with a student's outcome.

Check Grades: Look at the boxplots for 'Curricular units 2nd sem (grade)'. As you'd expect, students with higher grades are far more likely to graduate.

**Step 2:** Clean the Data (Preprocessing)
File: (Still in) notebooks/1_EDA_and_Preprocessing.ipynb

After the analysis, you'll see the data cleaning steps. Pay close attention to these:

Building a Pipeline: Look for the ColumnTransformer. This is the professional way to handle preprocessing. It automatically applies StandardScaler to all number columns and OneHotEncoder to all category columns.

Handling Imbalance (SMOTE): This is our solution to the imbalance problem. Look for the SMOTE code. You'll see it's only applied to the training data, which is a critical best practice.

**Step 3:** Train and Compare Models
File: notebooks/2_Model_Training.ipynb

Now, open the second notebook: notebooks/2_Model_Training.ipynb. This is where the machine learning happens.

Train 10 Models: Look at the models dictionary. You'll see we are training 10 different algorithms, from a simple LogisticRegression to a complex XGBClassifier.

Evaluate Performance: Run the training loop and look at the final results_df DataFrame. Your job is to compare the models based on their Weighted F1-Score. Don't just look at "Accuracy"—it's misleading because of the class imbalance.

Save the Best: The last cell saves the winning model to the models/ folder. This is the model your Streamlit app will use.

**Step 4:** Use the Final App
File: app/app.py

Finally, look at app/app.py. This is the code for the web app. You don't need to be a Streamlit expert; just look at the main() function:

It loads your three .joblib files from the models/ folder.

It creates the sidebar for you to enter data.

When you click "Predict," it uses your saved preprocessor and best_model to get a live prediction.



**How to Run This Project:**
Follow these instructions exactly to get the project running on your own machine.


**1. Setup Your Environment:**
First, clone this repository (or download the files).

Open a terminal and cd into the student-dropout-predictor folder.

Create your own virtual environment:

Bash
*python -m venv .venv*

Activate the environment.

On Windows: *.\.venv\Scripts\Activate*

On Mac/Linux: *source .venv/bin/activate*

Install all the required libraries at once:

Bash
*pip install -r requirements.txt*


**2. Run the Notebooks:**
You must run the notebooks first to create the model files. The app won't work without them.

Open the project folder in VSCode.

Open notebooks/1_EDA_and_Preprocessing.ipynb.

In the top-right corner, select your new .venv as the kernel.

Click the "Run All" button and wait for it to finish.

Now, open notebooks/2_Model_Training.ipynb.

Click the "Run All" button again and wait for it to finish.

Check your models/folder. You should now see three .joblib files inside it (best_model.joblib, label_encoder.joblib, preprocessor.joblib). If you do, you're ready!


3. **Launch the Streamlit App:**
Go back to your terminal (make sure your .venv is still active).

Run this command:

Bash
*streamlit run app/app.py*
Your web browser will automatically open. Go ahead and test your new app.



**Technologies You Will Use:**
Python 3.10+

Data Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Preprocessing: Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)

Imbalance Handling: imbalanced-learn (SMOTE)

Machine Learning: Scikit-learn, XGBoost, LightGBM, CatBoost

Web App: Streamlit

Environment: Jupyter Notebook, VSCode
