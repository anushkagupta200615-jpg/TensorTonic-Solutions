import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    
    num_states = len(values)
    num_actions = len(transitions[0])
    new_values = []

    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            expected_future = sum(transitions[s][a][s_prime] * values[s_prime]
                                  for s_prime in range(num_states))
            q = rewards[s][a] + gamma * expected_future
            q_values.append(q)
        new_values.append(max(q_values))
    
    return new_values
