from collections import deque


class Node:
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.observed = False


def create_bayesian_network():
    # Create nodes
    nodes = {
        "Study Habits": Node("Study Habits"),
        "Motivation": Node("Motivation"),
        "Learning Resources": Node("Learning Resources"),
        "Understanding of Concepts": Node("Understanding of Concepts"),
        "Assignments": Node("Assignments"),
        "Exam Anxiety": Node("Exam Anxiety"),
        "Exam Performance": Node("Exam Performance")
    }

    # Establish relationships
    nodes["Study Habits"].children.append(nodes["Understanding of Concepts"])
    nodes["Motivation"].children.append(nodes["Understanding of Concepts"])
    nodes["Learning Resources"].children.append(nodes["Understanding of Concepts"])
    nodes["Understanding of Concepts"].parents.extend(
        [nodes["Study Habits"], nodes["Motivation"], nodes["Learning Resources"]])

    nodes["Understanding of Concepts"].children.extend([nodes["Assignments"], nodes["Exam Performance"]])
    nodes["Assignments"].parents.append(nodes["Understanding of Concepts"])
    nodes["Exam Performance"].parents.extend([nodes["Understanding of Concepts"], nodes["Exam Anxiety"]])

    nodes["Exam Anxiety"].children.append(nodes["Exam Performance"])

    return nodes


def bayes_ball(start, target, nodes, observed_nodes):
    queue = deque([(start, "down")])  # Start ball from the start node in the down direction
    visited = set()

    # Mark observed nodes
    for obs in observed_nodes:
        nodes[obs].observed = True

    while queue:
        current, direction = queue.popleft()

        if (current.name, direction) in visited:
            continue
        visited.add((current.name, direction))

        # If we reach the target node, return False (they are dependent)
        if current == target:
            return False

        # Propagate the ball
        if direction == "down":
            # Move to children (if current node is not observed)
            if not current.observed:
                for child in current.children:
                    queue.append((child, "down"))
            # Move to parents
            for parent in current.parents:
                queue.append((parent, "up"))

        elif direction == "up":
            # If current is a collider and observed, continue propagating
            if current.observed:
                for child in current.children:
                    queue.append((child, "down"))
            # Move to parents
            if not current.observed:
                for parent in current.parents:
                    queue.append((parent, "up"))

    # If we can't reach the target node, return True (they are independent)
    return True


def main():
    # Create the Bayesian network
    nodes = create_bayesian_network()

    # Set "Exam Performance" as observed
    observed_nodes = ["Exam Performance"]

    # Check if Study Habits and Motivation are independent using Bayes-Ball algorithm
    independent = bayes_ball(nodes["Study Habits"], nodes["Motivation"], nodes, observed_nodes)

    # Output the result
    if independent:
        print("Study Habits and Motivation are independent given Exam Performance.")
    else:
        print("Study Habits and Motivation are dependent given Exam Performance.")


if __name__ == "__main__":
    main()
