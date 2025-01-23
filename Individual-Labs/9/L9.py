import random

def read_config_file(filename="config.txt"):
    """Reads configuration from file, default config.txt."""
    config = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=')
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Configuration file '{filename}' not found. Using default values.")
        return None

    n = int(config.get('N', 3))
    c = int(config.get('c', 1))
    l = int(config.get('l', 2))
    k = int(config.get('k', 3))
    return n, (c, l), k

def get_initial_state(n):
    """Returns initial robot and obstacle positions and probabilities."""
    initial_robot_pos = (n // 2 + 1, n // 2 + 1) # Middle of the table, integer division handles both odd and even N
    initial_obstacle_pos = (n, n) # Bottom-right corner as default if not specified
    robot_prob = {initial_robot_pos: 1.0}
    obstacle_prob = {initial_obstacle_pos: 1.0}
    return robot_prob, obstacle_prob, initial_robot_pos, initial_obstacle_pos

def get_robot_next_pos_prob(current_robot_pos, action, n):
    """Returns probability distribution of next robot positions given current position and action."""
    prob_distribution = {}
    row, col = current_robot_pos
    possible_next_positions = []

    if action == 'up':
        next_pos = (row - 1, col)
    elif action == 'down':
        next_pos = (row + 1, col)
    elif action == 'left':
        next_pos = (row, col - 1)
    elif action == 'right':
        next_pos = (row, col + 1)
    else: # stay
        next_pos = (row, col)

    # Success move (0.8 probability)
    if 1 <= next_pos[0] <= n and 1 <= next_pos[1] <= n: # Check grid boundaries
        prob_distribution[next_pos] = prob_distribution.get(next_pos, 0) + 0.8
    else:
        prob_distribution[current_robot_pos] = prob_distribution.get(current_robot_pos, 0) + 0.8 # Stay in place if move out of bounds

    # Failure to move (0.2 probability) - stay in the same position
    prob_distribution[current_robot_pos] = prob_distribution.get(current_robot_pos, 0) + 0.2

    return prob_distribution

def get_obstacle_next_pos_prob(current_obstacle_pos, n):
    """Returns probability distribution of next obstacle positions given current position."""
    prob_distribution = {}
    row, col = current_obstacle_pos
    adjacent_positions = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0,0)]: # right, left, down, up, stay
        next_pos = (row + dr, col + dc)
        if 1 <= next_pos[0] <= n and 1 <= next_pos[1] <= n: # Check grid boundaries
            adjacent_positions.append(next_pos)

    prob = 1.0 / len(adjacent_positions) if adjacent_positions else 0

    for pos in adjacent_positions:
        prob_distribution[pos] = prob_distribution.get(pos, 0) + prob

    return prob_distribution

def get_sensor_prob(robot_pos, obstacle_pos):
    """Returns probability distribution of sensor measurements given robot and obstacle positions."""
    sensor_prob = {}
    if robot_pos == (2,2) and obstacle_pos == (2,3):
        sensor_prob['close'] = 0.7
        sensor_prob['far'] = 0.3
    elif robot_pos == (2,2) and obstacle_pos == (3,3):
        sensor_prob['close'] = 0.3
        sensor_prob['far'] = 0.7
    elif robot_pos == (1,1) and obstacle_pos == (3,3):
        sensor_prob['far'] = 0.9
        sensor_prob['close'] = 0.1
    else: # Default case if no example matches, assuming "far" if not specifically "close" cases
        sensor_prob['far'] = 0.9
        sensor_prob['close'] = 0.1
    return sensor_prob

def forward_propagation(n, initial_robot_prob, initial_obstacle_prob, actions, observations):
    """Performs forward propagation for k time steps."""
    robot_prob_dist = initial_robot_prob
    obstacle_prob_dist = initial_obstacle_prob

    for t in range(len(actions)):
        action = actions[t]
        observation = observations[t]

        next_robot_prob_dist = {}
        next_obstacle_prob_dist = {}

        # Predict robot position
        for current_robot_pos, robot_prob in robot_prob_dist.items():
            robot_next_pos_probs = get_robot_next_pos_prob(current_robot_pos, action, n)
            for next_robot_pos, prob in robot_next_pos_probs.items():
                next_robot_prob_dist[next_robot_pos] = next_robot_prob_dist.get(next_robot_pos, 0) + robot_prob * prob

        # Predict obstacle position
        for current_obstacle_pos, obstacle_prob in obstacle_prob_dist.items():
            obstacle_next_pos_probs = get_obstacle_next_pos_prob(current_obstacle_pos, n)
            for next_obstacle_pos, prob in obstacle_next_pos_probs.items():
                next_obstacle_prob_dist[next_obstacle_pos] = next_obstacle_prob_dist.get(next_obstacle_pos, 0) + obstacle_prob * prob

        # Incorporate sensor measurement (Observation - Zt) - In this simplified version, we are asked for probability *before* incorporating sensor, so skipping observation update.
        robot_prob_dist = next_robot_prob_dist
        obstacle_prob_dist = next_obstacle_prob_dist

        # Normalize probabilities (important after several steps)
        robot_prob_norm_factor = sum(robot_prob_dist.values())
        if robot_prob_norm_factor > 0:
            robot_prob_dist = {pos: prob / robot_prob_norm_factor for pos, prob in robot_prob_dist.items()}

        obstacle_prob_norm_factor = sum(obstacle_prob_dist.values())
        if obstacle_prob_norm_factor > 0:
            obstacle_prob_dist = {pos: prob / obstacle_prob_norm_factor for pos, prob in obstacle_prob_dist.items()}


    return robot_prob_dist, obstacle_prob_dist

if __name__ == "__main__":
    config_data = read_config_file()
    if config_data:
        n, target_pos, k = config_data
    else:
        n, target_pos, k = 3, (1, 2), 3 # Default values if config file not found

    initial_robot_prob, initial_obstacle_prob, initial_robot_pos, initial_obstacle_pos = get_initial_state(n)

    print(f"Grid size: {n}x{n}")
    print(f"Initial robot position: {initial_robot_pos}")
    print(f"Initial obstacle position: {initial_obstacle_pos}")
    print(f"Target robot position: {target_pos}")
    print(f"Time steps: {k}")

    # Example actions (replace with desired sequence of actions for k steps)
    actions_sequence = ['up'] * k # Example: Robot tries to move up for k steps
    observations_sequence = ['none'] * k # We are not using observations in this simplified forward propagation for prediction.

    final_robot_prob_dist, final_obstacle_prob_dist = forward_propagation(n, initial_robot_prob, initial_obstacle_prob, actions_sequence, observations_sequence)

    probability_at_target = final_robot_prob_dist.get(target_pos, 0.0)

    print(f"\nProbability distribution of robot position after {k} time steps:")
    for pos, prob in sorted(final_robot_prob_dist.items()):
        print(f"Position: {pos}, Probability: {prob:.4f}")

    print(f"\nProbability distribution of obstacle position after {k} time steps:")
    for pos, prob in sorted(final_obstacle_prob_dist.items()):
        print(f"Position: {pos}, Probability: {prob:.4f}")


    print(f"\nProbability of robot being at position {target_pos} after {k} time steps: {probability_at_target:.4f}")