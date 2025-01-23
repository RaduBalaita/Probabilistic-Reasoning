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
        "Battery Failure": Node("Battery Failure"),
        "Starter Motor Failure": Node("Starter Motor Failure"),
        "Ignition System Failure": Node("Ignition System Failure"),
        "Fuel System Failure": Node("Fuel System Failure"),
        "Engine Failure": Node("Engine Failure"),
        "Not Start": Node("Not Start")
    }

    # Establish relationships
    nodes["Battery Failure"].children.extend([nodes["Starter Motor Failure"], nodes["Ignition System Failure"]])
    nodes["Starter Motor Failure"].parents.append(nodes["Battery Failure"])
    nodes["Ignition System Failure"].parents.append(nodes["Battery Failure"])

    nodes["Starter Motor Failure"].children.append(nodes["Engine Failure"])
    nodes["Ignition System Failure"].children.append(nodes["Engine Failure"])
    nodes["Fuel System Failure"].children.append(nodes["Engine Failure"])
    nodes["Engine Failure"].parents.extend(
        [nodes["Starter Motor Failure"], nodes["Ignition System Failure"], nodes["Fuel System Failure"]])

    nodes["Engine Failure"].children.append(nodes["Not Start"])
    nodes["Not Start"].parents.append(nodes["Engine Failure"])

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

    # Set "Engine Failure" as observed
    observed_nodes = ["Engine Failure"]

    # Check if Battery Failure and Fuel System Failure are independent using Bayes-Ball algorithm
    independent = bayes_ball(nodes["Battery Failure"], nodes["Fuel System Failure"], nodes, observed_nodes)

    # Output the result
    if independent:
        print("Battery Failure and Fuel System Failure are independent given Engine Failure.")
    else:
        print("Battery Failure and Fuel System Failure are dependent given Engine Failure.")


if __name__ == "__main__":
    main()
