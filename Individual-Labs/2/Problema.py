def rare_disease_test():
    p_disease = 1 / 10000
    p_positive_given_disease = 0.99
    p_positive_given_no_disease = 0.01

    # P(Positive)
    p_positive = (p_positive_given_disease * p_disease) + \
                 (p_positive_given_no_disease * (1 - p_disease))

    # P(Disease|Positive)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    return round(p_disease_given_positive * 100, 2)  # Return percentage
