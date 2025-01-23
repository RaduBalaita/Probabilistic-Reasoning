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
        "Network Switch Failure": Node("Network Switch Failure"),
        "Server Failure": Node("Server Failure"),
        "Database Failure": Node("Database Failure"),
        "Application Failure": Node("Application Failure")
    }

    # Establish relationships
    nodes["Network Switch Failure"].children.extend([nodes["Server Failure"], nodes["Database Failure"]])
    nodes["Server Failure"].parents.append(nodes["Network Switch Failure"])
    nodes["Database Failure"].parents.append(nodes["Network Switch Failure"])

    nodes["Server Failure"].children.append(nodes["Application Failure"])
    nodes["Database Failure"].children.append(nodes["Application Failure"])
    nodes["Application Failure"].parents.extend([nodes["Server Failure"], nodes["Database Failure"]])

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

    # Set "Application Failure" as observed
    observed_nodes = ["Application Failure"]

    # Check if Server Failure and Database Failure are independent using Bayes-Ball algorithm
    independent = bayes_ball(nodes["Server Failure"], nodes["Database Failure"], nodes, observed_nodes)

    # Output the result
    if independent:
        print("Server Failure and Database Failure are independent given Application Failure.")
    else:
        print("Server Failure and Database Failure are dependent given Application Failure.")


if __name__ == "__main__":
    main()
