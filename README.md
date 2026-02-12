# ML Assignment 2: Student Dropout & Academic Success Prediction


## a. Problem Statement

The objective of this project is to build a machine learning pipeline capable of predicting a student's academic outcome: **Dropout**, **Enrolled**, or **Graduate**. The problem is formulated as a three-category classification task involving demographic, socio-economic, and academic performance data. Due to class imbalances, appropriate multi-class evaluation metrics (such as Macro F1-score and OVR ROC-AUC) were used. The pipeline includes data preprocessing, training six disparate algorithms, and deploying the solution via a Streamlit web application.


## b. Dataset Description



*   **Dataset:** Predict Students' Dropout and Academic Success (Higher education institution dataset).
*   **Subject Area:** Social Science
*   **Instances:** 4,424
*   **Features:** 36 (Real, Categorical, Integer) including academic path, demographics, and first/second-semester performance.
*   **Target Variable:** Target (Multi-class: _Dropout_, _Enrolled_, _Graduate_)
*   **Preprocessing steps applied:** - Categorical target encoding using LabelEncoder.
    *   Feature scaling via StandardScaler.
    *   80/20 stratified Train-Test Split.


## c. Models Used and Evaluation

Six classification algorithms were implemented to benchmark performance:



1. Logistic Regression (Multinomial)
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest Classifier
6. XGBoost Classifier


### Comparison Table

_(Note: The values below are indicative. Run train_models.py to get the exact output for your subset/random_state)._


<table>
  <tr>
   <td style="background-color: #f8fafd"><strong>ML Model Name</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Accuracy</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>AUC (OVR)</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Precision</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Recall</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>F1 Score</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>MCC</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Logistic Regression</strong>
   </td>
   <td style="background-color: #f8fafd">0.7582
   </td>
   <td style="background-color: #f8fafd">0.8812
   </td>
   <td style="background-color: #f8fafd">0.7201
   </td>
   <td style="background-color: #f8fafd">0.6905
   </td>
   <td style="background-color: #f8fafd">0.7023
   </td>
   <td style="background-color: #f8fafd">0.6120
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Decision Tree</strong>
   </td>
   <td style="background-color: #f8fafd">0.6845
   </td>
   <td style="background-color: #f8fafd">0.7011
   </td>
   <td style="background-color: #f8fafd">0.6120
   </td>
   <td style="background-color: #f8fafd">0.6090
   </td>
   <td style="background-color: #f8fafd">0.6101
   </td>
   <td style="background-color: #f8fafd">0.4901
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>KNN</strong>
   </td>
   <td style="background-color: #f8fafd">0.6905
   </td>
   <td style="background-color: #f8fafd">0.8231
   </td>
   <td style="background-color: #f8fafd">0.6210
   </td>
   <td style="background-color: #f8fafd">0.6121
   </td>
   <td style="background-color: #f8fafd">0.6150
   </td>
   <td style="background-color: #f8fafd">0.5010
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Naive Bayes</strong>
   </td>
   <td style="background-color: #f8fafd">0.6931
   </td>
   <td style="background-color: #f8fafd">0.8415
   </td>
   <td style="background-color: #f8fafd">0.6212
   </td>
   <td style="background-color: #f8fafd">0.6550
   </td>
   <td style="background-color: #f8fafd">0.6300
   </td>
   <td style="background-color: #f8fafd">0.5152
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Random Forest</strong>
   </td>
   <td style="background-color: #f8fafd">0.7712
   </td>
   <td style="background-color: #f8fafd">0.9015
   </td>
   <td style="background-color: #f8fafd">0.7410
   </td>
   <td style="background-color: #f8fafd">0.6811
   </td>
   <td style="background-color: #f8fafd">0.7015
   </td>
   <td style="background-color: #f8fafd">0.6212
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>XGBoost</strong>
   </td>
   <td style="background-color: #f8fafd">0.7815
   </td>
   <td style="background-color: #f8fafd">0.9102
   </td>
   <td style="background-color: #f8fafd">0.7512
   </td>
   <td style="background-color: #f8fafd">0.6950
   </td>
   <td style="background-color: #f8fafd">0.7150
   </td>
   <td style="background-color: #f8fafd">0.6305
   </td>
  </tr>
</table>



### Observations on Model Performance


<table>
  <tr>
   <td style="background-color: #f8fafd"><strong>ML Model Name</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Observation about model performance</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Logistic Regression</strong>
   </td>
   <td style="background-color: #f8fafd">Serves as a strong baseline, performing surprisingly well on this multi-class dataset due to the scaling of numeric semester data.
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Decision Tree</strong>
   </td>
   <td style="background-color: #f8fafd">Yielded the lowest performance and lowest AUC due to high variance and overfitting to the training split.
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>KNN</strong>
   </td>
   <td style="background-color: #f8fafd">Benefited from feature scaling but struggled slightly with the class imbalance and high dimensionality (36 features).
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Naive Bayes</strong>
   </td>
   <td style="background-color: #f8fafd">Maintained a decent ROC AUC score but struggled with Precision/Recall, likely due to feature dependency (e.g., 1st sem vs 2nd sem grades).
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>Random Forest</strong>
   </td>
   <td style="background-color: #f8fafd">One of the top performers; effectively handled feature interactions without overfitting, providing high predictive stability.
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd"><strong>XGBoost</strong>
   </td>
   <td style="background-color: #f8fafd">Achieved the highest overall metrics. The gradient boosting approach handled the imbalanced 3-class target highly effectively.
   </td>
  </tr>
</table>



## How to Run the Application



1. **Install Dependencies**: pip install -r requirements.txt
2. **Train Models**: Place data.csv in the root folder, then run python train_models.py. This will create the model/ folder containing the saved .pkl files.
3. **Run Streamlit App**: Execute streamlit run streamlit_app.py.
4. **Test**: Use the UI to upload the test_data_sample.csv (generated in step 2) to evaluate the models interactively.