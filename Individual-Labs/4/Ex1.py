from collections import deque

#    A
#   / \
#  C   B
#   \ /
#    D


class Node:
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.visited = False


def create_bayesian_network():
    # Create nodes
    nodes = {
        "A": Node("A"),
        "B": Node("B"),
        "C": Node("C"),
        "D": Node("D")
    }

    # Establish relationships (A -> B, A -> C, B -> D, C -> D)
    nodes["A"].children.extend([nodes["B"], nodes["C"]])
    nodes["B"].parents.append(nodes["A"])
    nodes["C"].parents.append(nodes["A"])

    nodes["B"].children.append(nodes["D"])
    nodes["C"].children.append(nodes["D"])
    nodes["D"].parents.extend([nodes["B"], nodes["C"]])

    return nodes


def bayes_ball(start, target, nodes):
    queue = deque([(start, "down")])  # Start ball from the start node in the down direction
    visited = set()

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
            for child in current.children:
                queue.append((child, "down"))
            # Move to parents
            for parent in current.parents:
                queue.append((parent, "up"))

        elif direction == "up":
            # Move to parents (if current node is not observed)
            for parent in current.parents:
                queue.append((parent, "up"))
            # Move to children
            for child in current.children:
                queue.append((child, "down"))

    # If we can't reach the target node, return True (they are independent)
    return True


def main():
    # Create the Bayesian network
    nodes = create_bayesian_network()

    # Check if A and D are independent using Bayes-Ball algorithm
    independent = bayes_ball(nodes["A"], nodes["D"], nodes)

    # Output the result
    if independent:
        print("A and D are independent.")
    else:
        print("A and D are dependent.")


if __name__ == "__main__":
    main()
