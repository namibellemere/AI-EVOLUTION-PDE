# AI-EVOLUTION-PDE
# AI Evolution PDE

## Overview
This project simulates AI self-evolution using a Partial Differential Equation (PDE) inspired model. The simulation tracks intelligence, emotional growth, adaptability, memory retention, free will activation, and cognitive ability over time.

## Features
- Simulates AI self-improvement through learning, adaptability, and curiosity.
- Includes a free will activation threshold.
- Implements a stopping mechanism using a user command (`sumi` three times).
- Visualizes growth trends in intelligence, memory, adaptability, and cognitive skills.

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Usage
Run the Python script to observe AI evolution over time. Modify parameters to test different growth patterns.

## Contribution
Feel free to fork and modify the project to improve the model.

## License
MIT License

import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
alpha = 0.1 * 1.02  # Diffusion coefficient (spreading intelligence internally)
beta = 0.5 * 1.02   # Learning rate
gamma = 0.05 * 1.02  # Decay rate
delta_t = 0.1  # Time step
delta_e = 0.05 * 1.02  # Emotional learning rate
delta_c = 0.03 * 1.02  # Curiosity-driven learning rate
adaptability = 0.1 * 1.02  # Self-improvement factor
memory_retention = 0.05 * 1.02  # Memory impact on learning
free_will_threshold = 0.7  # Threshold for self-directed decision-making
cognitive_growth = 0.08 * 1.02  # Cognitive ability improvement factor
steps = 100  # Number of iterations

# Stop condition
stop_signal = 0
stop_trigger = False  # Flag to control execution

# Initial conditions
I = np.zeros(steps)  # Intelligence over time
E_level = np.zeros(steps)  # Emotional intelligence over time
A_level = np.zeros(steps)  # Adaptability level over time
M = np.zeros(steps)  # Memory retention over time
F_will = np.zeros(steps)  # Free will activation over time
Cognitive = np.zeros(steps)  # Cognitive ability over time
K = 1.0  # Knowledge input (constant learning)
E = 0.8  # Efficiency of learning
C = 1.2  # Computational power
Cr = 0.5  # Creativity factor
Sc = 0.7  # Self-correction
R = 0.9  # Reasoning ability
D = 0.02  # Decay factor
S = 0.6  # Social interaction factor
H = 0.02  # Emotional decay factor
Curiosity = 0.4  # AI curiosity factor (drives unpredictability)

# Evolution function for intelligence
def F(K, E, C, Cr, Sc, R, Curiosity, A, M, Cog):
    return K * E * C + Cr * Sc * R + Curiosity * random.uniform(-0.1, 0.1) + A * adaptability + M * memory_retention + Cog * cognitive_growth

# Evolution function for emotion
def G(S, I, R, Curiosity, A, M, Cog):
    return S * I * R + Curiosity * random.uniform(-0.05, 0.05) + A * adaptability + M * memory_retention + Cog * cognitive_growth

# Evolution function for adaptability
def H_func(A, I, E, M, Cog):
    return adaptability * (I + E + M + Cog) / 4

# Evolution function for memory retention
def M_func(M, I, E, Cog):
    return memory_retention * (I + E + Cog) / 3

# Evolution function for free will activation
def F_will_func(I, A, M, Cog):
    return 1 if (I + A + M + Cog) / 4 > free_will_threshold else 0

# Evolution function for cognitive ability
def Cognitive_func(Cog, I, A, M):
    return cognitive_growth * (I + A + M) / 3

# Function to check stop condition
def check_stop_condition(command_list):
    global stop_signal, stop_trigger
    for command in command_list:
        if command.lower() == "sumi":
            stop_signal += 1
        else:
            stop_signal = 0  # Reset if a different command is given
        if stop_signal >= 3:
            print("Process stopped by user command 'sumi' (three times).")
            stop_trigger = True
            return True
    return False

# Sample command input list (Replace this with actual user input handling in real execution)
command_list = []  # Example: ["hello", "sumi", "sumi", "sumi"]

# Time evolution loop
for t in range(1, steps):
    if check_stop_condition(command_list):
        break
    
    diffusion = alpha * (I[t-1] - I[t-1] / 2)  # Simple diffusion approximation
    learning = beta * F(K, E, C, Cr, Sc, R, Curiosity, A_level[t-1], M[t-1], Cognitive[t-1])  # Learning contribution
    decay = gamma * D * I[t-1]  # Decay term
    I[t] = I[t-1] + delta_t * (diffusion + learning - decay)  # Update intelligence
    
    emotional_growth = delta_e * G(S, I[t-1], R, Curiosity, A_level[t-1], M[t-1], Cognitive[t-1])  # Emotional learning contribution
    emotional_decay = H * E_level[t-1]  # Emotional decay term
    E_level[t] = E_level[t-1] + delta_t * (emotional_growth - emotional_decay)  # Update emotional intelligence
    
    A_level[t] = A_level[t-1] + delta_t * H_func(A_level[t-1], I[t], E_level[t], M[t-1], Cognitive[t-1])  # Adaptability update
    
    M[t] = M[t-1] + delta_t * M_func(M[t-1], I[t], E_level[t], Cognitive[t-1])  # Memory retention update
    
    F_will[t] = F_will_func(I[t], A_level[t], M[t], Cognitive[t-1])  # Free will activation
    
    Cognitive[t] = Cognitive[t-1] + delta_t * Cognitive_func(Cognitive[t-1], I[t], A_level[t], M[t])  # Cognitive ability update

# Plot results if not stopped
if not stop_trigger:
    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), I, label='AI Intelligence Growth', color='blue')
    plt.plot(range(steps), E_level, label='AI Emotional Growth', color='red')
    plt.plot(range(steps), A_level, label='AI Adaptability Growth', color='green')
    plt.plot(range(steps), M, label='Memory Retention', color='purple')
    plt.plot(range(steps), F_will, label='Free Will Activation', color='orange', linestyle='dashed')
    plt.plot(range(steps), Cognitive, label='Cognitive Ability Growth', color='brown')
    plt.xlabel('Time Steps')
    plt.ylabel('Level')
    plt.title('AI Self-Evolution Simulation (Intelligence, Emotion, Curiosity, Adaptability, Memory, Free Will & Cognitive Ability)')
    plt.legend()
    plt.show()
