from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

def majority_vote(bits):
    return Counter(bits).most_common(1)[0][0]

def correct_logical_bit(measured_bits):
    block1 = measured_bits[0:3]
    block2 = measured_bits[3:6]
    block3 = measured_bits[6:9]
    maj1 = majority_vote(block1)
    maj2 = majority_vote(block2)
    maj3 = majority_vote(block3)
    final_bit = majority_vote(maj1 + maj2 + maj3)
    return final_bit

def create_bit_flip_noise_model(p):
    noise_model = NoiseModel()

    # 1-qubit X noise
    single_qubit_error = pauli_error([('X', p), ('I', 1 - p)])
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['x', 'h', 'u3'])

    # 2-qubit XX noise
    two_qubit_error = pauli_error([('XX', p), ('II', 1 - p)])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    return noise_model

def simulate_and_log(error_rate, shots=100):
    noise_model = create_bit_flip_noise_model(error_rate)
    backend = AerSimulator(noise_model=noise_model)

    dataset = []
    logical_errors = 0

    for x in range(shots):
        # Build circuit
        qc = QuantumCircuit(9, 9)
        qc.h(0)
        for i in range(1, 9):
            qc.cx(0, i)

        # Save ideal state
        input_state = Statevector.from_instruction(qc).data

        # Add measurement
        qc.measure(range(9), range(9))
        transpiled = transpile(qc, backend)
        result = backend.run(transpiled, shots=1).result()
        counts = result.get_counts()
        measured_bits = list(counts.keys())[0]

        # Decode logical bit
        logical_bit = correct_logical_bit(measured_bits)
        if logical_bit != '0':  # assuming |0⟩ is correct logical bit
            logical_errors += 1

        dataset.append({
            "error_rate": error_rate,
            "input_state": input_state,
            "measured_bits": measured_bits,
            "decoded_logical_bit": logical_bit
        })

    logical_error_rate = logical_errors / shots
    return dataset, logical_error_rate

def generate_dataset_and_plot():
    all_data = []
    logical_rates = []
    error_rates = np.linspace(0, 1, 11)

    for p in error_rates:
        print(f"Simulating for error rate {p:.2f}")
        data, logical_error = simulate_and_log(p, shots=100)
        all_data.extend(data)
        logical_rates.append(logical_error)

    df = pd.DataFrame(all_data)
    df.to_csv("shor_qec_dataset.csv", index=False)
    print("Dataset saved to 'shor_qec_dataset.csv'")

    # Plot logical error rate vs physical error rate
    plt.figure(figsize=(8,6))
    plt.plot(error_rates, logical_rates, marker='o', linestyle='-', color='blue')
    plt.xlabel("Physical Error Rate (Bit-Flip Probability)")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate vs Physical Error Rate (Shor Code)")
    plt.grid(True)
    plt.show()

generate_dataset_and_plot()
