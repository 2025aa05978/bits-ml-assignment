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


<table>
  <tr>
   <td><strong>ML Model Name</strong>
   </td>
   <td><strong>Accuracy</strong>
   </td>
   <td><strong>AUC (OVR)</strong>
   </td>
   <td><strong>Precision</strong>
   </td>
   <td><strong>Recall</strong>
   </td>
   <td><strong>F1 Score</strong>
   </td>
   <td><strong>MCC</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Logistic Regression</strong>
   </td>
   <td>0.7722
   </td>
   <td>0.8884
   </td>
   <td>0.7142
   </td>
   <td>0.6819
   </td>
   <td>0.6902
   </td>
   <td>0.6216
   </td>
  </tr>
  <tr>
   <td><strong>Decision Tree</strong>
   </td>
   <td>0.9344
   </td>
   <td>0.9451
   </td>
   <td>0.9205
   </td>
   <td>0.9237
   </td>
   <td>0.9220
   </td>
   <td>0.8937
   </td>
  </tr>
  <tr>
   <td><strong>KNN</strong>
   </td>
   <td>0.7706
   </td>
   <td>0.8904
   </td>
   <td>0.7391
   </td>
   <td>0.6872
   </td>
   <td>0.7018
   </td>
   <td>0.6190
   </td>
  </tr>
  <tr>
   <td><strong>Naive Bayes</strong>
   </td>
   <td>0.6792
   </td>
   <td>0.7992
   </td>
   <td>0.5892
   </td>
   <td>0.5786
   </td>
   <td>0.5749
   </td>
   <td>0.4618
   </td>
  </tr>
  <tr>
   <td><strong>Random Forest</strong>
   </td>
   <td>0.9530
   </td>
   <td>0.9904
   </td>
   <td>0.9520
   </td>
   <td>0.9345
   </td>
   <td>0.9426
   </td>
   <td>0.9234
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost</strong>
   </td>
   <td>0.9514
   </td>
   <td>0.9871
   </td>
   <td>0.9455
   </td>
   <td>0.9357
   </td>
   <td>0.9404
   </td>
   <td>0.9208
   </td>
  </tr>
</table>



### Observations on Model Performance


<table>
  <tr>
   <td><strong>ML Model Name</strong>
   </td>
   <td><strong>Observation about model performance</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Logistic Regression</strong>
   </td>
   <td>Performed moderately well (77% accuracy) and acts as a solid baseline, but struggles to capture more complex, non-linear relationships compared to tree-based models.
   </td>
  </tr>
  <tr>
   <td><strong>Decision Tree</strong>
   </td>
   <td>Showed excellent performance (~93% accuracy). The data contains strong, hierarchical decision boundaries (e.g., specific grade thresholds) that single trees capture highly effectively.
   </td>
  </tr>
  <tr>
   <td><strong>KNN</strong>
   </td>
   <td>Performed similarly to Logistic Regression (77% accuracy). It handles the feature space well after standardization but falls short of the precision offered by ensemble techniques.
   </td>
  </tr>
  <tr>
   <td><strong>Naive Bayes</strong>
   </td>
   <td>The weakest performer on this dataset (~67% accuracy). The algorithm's assumption of feature independence heavily hurts its performance here, as features like 1st-semester and 2nd-semester grades are strongly correlated.
   </td>
  </tr>
  <tr>
   <td><strong>Random Forest</strong>
   </td>
   <td>Achieved the highest accuracy (95%) and a near-perfect AUC (0.9904). It successfully leveraged feature bagging to eliminate variance while easily mapping the complex relationships.
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost</strong>
   </td>
   <td>Good performance, almost on par with Random Forest. The gradient boosting approach handled the imbalanced 3-class target highly efficiently, resulting in a very high MCC score (0.9208).
   </td>
  </tr>
</table>

