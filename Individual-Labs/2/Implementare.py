from collections import defaultdict, Counter


class NaiveBayesSpamFilter:
    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)
        self.spam_emails = 0
        self.ham_emails = 0
        self.vocabulary = set()

    def train(self, emails, labels):
        for email, label in zip(emails, labels):
            words = email.lower().split()

            if label == 'spam':
                self.spam_emails += 1
                for word in words:
                    self.spam_word_counts[word] += 1
            else:
                self.ham_emails += 1
                for word in words:
                    self.ham_word_counts[word] += 1

            self.vocabulary.update(words)

    def calculate_word_probability(self, word, is_spam):
        """Calculate P(word|spam) or P(word|ham) with Laplace smoothing"""
        if is_spam:
            return (self.spam_word_counts[word] + 1) / (self.spam_emails + 2)
        return (self.ham_word_counts[word] + 1) / (self.ham_emails + 2)

    def classify(self, email):
        """
        Classify a new email as spam or not spam

        Args:
            email (str): The email text to classify

        Returns:
            tuple: (classification, spam_probability)
        """
        words = email.lower().split()

        # Calculate P(spam) and P(ham)
        total_emails = self.spam_emails + self.ham_emails
        p_spam = self.spam_emails / total_emails
        p_ham = self.ham_emails / total_emails

        # Calculate P(words|spam) and P(words|ham)
        spam_probability = p_spam
        ham_probability = p_ham

        for word in words:
            if word in self.vocabulary:
                spam_probability *= self.calculate_word_probability(word, True)
                ham_probability *= self.calculate_word_probability(word, False)

        # Normalize probabilities
        total_probability = spam_probability + ham_probability
        spam_probability = spam_probability / total_probability

        return ('spam' if spam_probability > 0.5 else 'not spam', spam_probability)


# Example usage
def main():
    # Training data
    training_emails = [
        "win free offer money",
        "meeting schedule tomorrow project",
        "free money win big"
    ]
    training_labels = ['spam', 'not spam', 'spam']

    # Create and train the spam filter
    spam_filter = NaiveBayesSpamFilter()
    spam_filter.train(training_emails, training_labels)

    # Test new email
    new_email = "win free money"
    classification, probability = spam_filter.classify(new_email)

    print(f"Email: '{new_email}'")
    print(f"Classification: {classification}")
    print(f"Spam probability: {probability:.2%}")


if __name__ == "__main__":
    main()