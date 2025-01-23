# Probabilistic Reasoning Labs

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](PR-Labs.ipynb)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)

**Repository showcasing 11 labs from a *Probabilistic Reasoning* course**, implementing key concepts and algorithms in probabilistic AI. Designed to demonstrate hands-on experience with Bayesian networks, inference, and sampling methods.

---

## 🧪 Labs Overview

### **Lab 2: Probabilistic Reasoning**
**Objective**: Apply Bayes' theorem and conditional probability to practical problems.
- Solved the "Rare Disease Test" problem to understand Bayes' theorem in diagnostics.
- Implemented a Naive Bayes Spam Filter for email classification.
- **Key Skill**: Applying Bayes' theorem and conditional probability for reasoning under uncertainty.

### **Lab 3: Random Variables**
**Objective**: Calculate covariance to analyze relationships between random variables.
- Analyzed student data to calculate covariance between study hours & exam scores, and class attendance & exam scores.
- Used `pandas` and `numpy` for data manipulation and covariance calculation.
- **Key Skill**: Understanding and applying covariance in data analysis.

### **Lab 4: Bayes-Ball Algorithm**
**Objective**: Implement the Bayes-Ball algorithm to determine conditional independence in Bayesian Networks.
- Built a Bayesian Network structure using custom `Node` class.
- Applied Bayes-Ball algorithm to assess independence in scenarios like A & D independence, Cold & Allergy, Server & Database failure, Battery & Fuel failure, and Study Habits & Motivation.
- **Key Skill**: Using Bayes-Ball for d-separation and conditional independence analysis.

### **Lab 5: Bayesian Networks - Exact Inference**
**Objective**: Perform exact inference in Bayesian Networks using enumeration.
- Calculated probability of traffic congestion based on weather and events using a simplified Bayesian Network.
- Predicted student admission probabilities based on IQ, exam difficulty, and aptitude scores.
- **Key Skill**: Implementing enumeration inference for exact probability calculation.

### **Lab 6: The Junction Tree Algorithm**
**Objective**: Utilize the Junction Tree Algorithm for efficient inference in complex Bayesian Networks.
- Developed a Disaster Response Decision Support System (DR-DSS) using `pgmpy`.
- Modeled a Bayesian Network for disaster scenarios to predict emergency response time and evacuation needs.
- **Key Skill**: Applying Junction Tree Algorithm (using `pgmpy`) for efficient inference in larger networks.

### **Lab 7: MCMC**
**Objective**: Implement Markov Chain Monte Carlo (MCMC) for posterior distribution estimation.
- Used basic MCMC sampling to estimate the posterior distribution for output current (IOUT) in a circuit diagnosis scenario.
- **Key Skill**: Understanding and implementing basic MCMC for approximate inference.

### **Lab 8: Dynamic Bayesian Networks (DBNs)**
**Objective**: Model time-series data using Dynamic Bayesian Networks and the Viterbi algorithm.
- Modeled tree growth rings to infer temperature sequences using a DBN.
- Implemented the Viterbi algorithm to find the most likely temperature sequence and compute state probabilities.
- **Key Skill**: Applying DBNs and Viterbi algorithm for sequence analysis and hidden state inference in time-series data.

### **Lab 9: Dynamic Bayesian Networks (cont.)**
**Objective**: Implement forward propagation in a DBN for robot navigation with dynamic obstacles.
- Simulated robot navigation in a grid with obstacle avoidance using forward propagation.
- Calculated probability distributions for robot and obstacle positions over time.
- **Key Skill**: Implementing forward propagation in DBNs for prediction and state estimation over time.

### **Lab 10: Causal Networks**
**Objective**: Design and implement a Causal Bayesian Network for student performance prediction.
- Developed a CBN to predict student pass/fail outcomes based on study habits, attendance, mental health, and prior performance.
- Categorized students into risk groups based on predicted performance probabilities.
- **Key Skill**: Designing and utilizing Causal Bayesian Networks for predictive modeling and risk assessment.

### **Lab 11: Causal Networks & Do-Calculus**
**Objective**: Apply Do-Calculus to analyze causal effects in Bayesian Networks.
- Analyzed the causal effect of vaccination on disease symptoms using a simplified causal network.
- Implemented calculations to determine P(Symptoms|do(Vaccination=1)).
- **Key Skill**: Understanding and applying Do-Calculus for causal inference and intervention analysis.

### **Lab 12: EM/GMM**
**Objective**: Integrate Expectation-Maximization (EM) with Gaussian Mixture Models (GMM) for customer segmentation within a Bayesian Network framework.
- Segmented customers based on purchasing habits and website time using GMM.
- Computed the probability of belonging to a customer segment given covariates like age and income.
- **Key Skill**: Combining EM/GMM for clustering and probabilistic inference in customer segmentation.

---

## 🛠️ Tools & Technologies
- **Core**: Python, NumPy, Pandas, Jupyter
- **Probabilistic Modeling**: `pgmpy` for Bayesian Networks
- **Clustering**: `sklearn.mixture` for Gaussian Mixture Models
- **Statistical Analysis**: `scipy.stats`
- **Visualization**: `matplotlib.pyplot`

---

## 📚 Key Learnings
1. **Probabilistic Reasoning Fundamentals**: Hands-on application of Bayes' theorem, conditional probability, and random variables.
2. **Bayesian Network Implementation**: Building and utilizing Bayesian Networks for inference, conditional independence analysis, and prediction.
3. **Inference Algorithms**: Practical experience with Enumeration Inference, Junction Tree Algorithm, MCMC, and Viterbi algorithm.
4. **Causal Inference**: Understanding and applying Causal Networks and Do-Calculus for analyzing causal effects.
5. **Clustering with GMM and EM**: Using Gaussian Mixture Models and Expectation-Maximization for customer segmentation.
6. **Real-world Problem Solving**: Applying probabilistic reasoning techniques to practical problems like spam filtering, medical diagnosis, disaster response, and customer segmentation.
