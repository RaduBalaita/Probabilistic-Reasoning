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
        "Cold": Node("Cold"),
        "Allergy": Node("Allergy"),
        "Sneezing": Node("Sneezing"),
        "Fever": Node("Fever")
    }

    # Establish relationships (Cold -> Fever, Cold -> Sneezing, Allergy -> Sneezing)
    nodes["Cold"].children.extend([nodes["Fever"], nodes["Sneezing"]])
    nodes["Fever"].parents.append(nodes["Cold"])
    nodes["Sneezing"].parents.append(nodes["Cold"])

    nodes["Allergy"].children.append(nodes["Sneezing"])
    nodes["Sneezing"].parents.append(nodes["Allergy"])

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

    # Set "Sneezing" as observed
    observed_nodes = ["Sneezing"]

    # Check if Cold and Allergy are independent using Bayes-Ball algorithm
    independent = bayes_ball(nodes["Cold"], nodes["Allergy"], nodes, observed_nodes)

    # Output the result
    if independent:
        print("Cold and Allergy are independent given Sneezing.")
    else:
        print("Cold and Allergy are dependent given Sneezing.")


if __name__ == "__main__":
    main()
