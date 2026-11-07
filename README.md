
# MSCS 634 – Lab 2: KNN and Radius Neighbors Classification

## 1. Purpose of the Lab

The purpose of this lab was to explore, implement, and compare two machine learning classification algorithms: K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), using the Wine Dataset available in the Scikit-learn library.

The Wine Dataset contains 178 samples, three wine classes, and 13 chemical features. The objective was to understand how different parameter values affect classification accuracy and how the two algorithms perform when classifying wine samples.

The dataset was divided into 80% training data and 20% testing data. KNN was evaluated using K values of 1, 5, 11, 15, and 21, while Radius Neighbors was evaluated using radius values of 350, 400, 450, 500, 550, and 600.

Python libraries, including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, were used for data exploration, model development, evaluation, and visualization.

## 2. Key Insights from Accuracy Trends and Observations

### K-Nearest Neighbors (KNN)

The KNN classifier was tested using five different values of K: 1, 5, 11, 15, and 21.

The experiment demonstrated that changing the number of neighbors affects classification accuracy. Smaller K values make the model more sensitive to individual observations, while larger K values consider more neighboring samples when making predictions.

The accuracy visualization helped identify the K value that produced the best classification performance on the testing dataset.

### Radius Neighbors (RNN)

The Radius Neighbors classifier was evaluated using radius values of 350, 400, 450, 500, 550, and 600.

Unlike KNN, which uses a fixed number of neighbors, Radius Neighbors considers all training observations within a specified distance.

The experiment demonstrated that increasing the radius changes the number of neighboring observations considered during classification. The accuracy results helped identify the radius value that performed best on the testing dataset.

### Comparison of KNN and RNN

Both algorithms use distance measurements to classify data, but they differ in how they identify neighboring observations.

KNN may be preferable when a consistent number of neighboring observations is required. Radius Neighbors may be preferable when the distance between observations and the density of nearby samples are important.

The best-performing configurations were identified by comparing the highest accuracy achieved by each algorithm. The results provide insight into how parameter selection influences classification performance.

## 3. Challenges Faced and Decisions Made

One challenge during this lab was selecting appropriate parameter values for both classification algorithms. Different K and radius values produced different classification results, demonstrating the importance of parameter selection.

Another consideration was the difference in numerical scales among the 13 chemical features in the Wine Dataset. Since the assignment specified radius values between 350 and 600, the original feature values were retained for the Radius Neighbors experiment. This allowed the required radius values to be tested on the original distance scale. However, features with larger numerical ranges can have a greater influence on distance calculations.

Another challenge involved handling observations that did not have any training samples within the selected radius. The `outlier_label="most_frequent"` parameter was used in the Radius Neighbors classifier to handle these observations.

The dataset was also divided using stratified sampling to maintain the distribution of the three wine classes in both the training and testing sets.

## 4. Conclusion

This lab provided practical experience implementing and evaluating KNN and Radius Neighbors classification algorithms using Python and Scikit-learn.

The experiment demonstrated that classification accuracy is influenced by parameter selection and the characteristics of the dataset. The accuracy tables and visualizations provided a useful way to compare the performance of both algorithms.

Overall, KNN and Radius Neighbors are useful classification techniques, but selecting the appropriate algorithm depends on the dataset, the distance measurements, and the desired classification performance.

## 5. Repository Contents

The GitHub repository contains the following files:

- `MSCS_634_Lab_2.ipynb` – Jupyter Notebook containing the Python implementation, dataset exploration, model evaluation, accuracy results, visualizations, and discussion.
- `README.md` – Documentation explaining the purpose of the lab, key findings, challenges, decisions, and conclusions.

## 6. Technologies Used

- Python 3
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- GitHub
