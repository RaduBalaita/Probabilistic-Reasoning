import pandas as pd
import numpy as np

#Dataset
data = {
    'Student': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Study Hours (X)': [5, 3, 6, 4, 8, 5, 7, 6, 4, 8, 9, 5, 7, 6, 5, 6, 7, 8, 5, 7],
    'Class Attendance (Y)': [80, 70, 90, 60, 100, 80, 90, 80, 70, 60, 80, 80, 90, 80, 70, 80, 90, 100, 80, 85],
    'Exam Score (Z)': [75, 65, 85, 70, 95, 75, 85, 75, 65, 70, 80, 75, 85, 75, 65, 75, 85, 95, 75, 80]
}

#Dataframe
df = pd.DataFrame(data)

#Covariance Matrix between Study Hours and Exam Score
cov_matrix_XZ = np.cov(df['Study Hours (X)'], df['Exam Score (Z)'])

#Extract Covariance
cov_XZ = cov_matrix_XZ[0][1]

#Covariance Matrix between Class Attendance and Exam Score
cov_matrix_YZ = np.cov(df['Class Attendance (Y)'], df['Exam Score (Z)'])

#Extract Covariance
cov_YZ = cov_matrix_YZ[0][1]

#Print
print("Covariance Matrix between Study Hours and Exam Score", "\n" , cov_matrix_XZ, "\n")
print("Covariance between Study Hours and Exam Score", "\n",  cov_XZ, "\n")
print("Covariance Matrix between Class Attendance and Exam Score", "\n", cov_matrix_YZ, "\n")
print("Covariance between Class Attendance and Exam Score", "\n", cov_YZ, "\n")