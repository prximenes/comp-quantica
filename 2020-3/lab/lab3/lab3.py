import qiskit
from qiskit.circuit.library import Diagonal
import numpy as np
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector, plot_histogram

def inv_media(vetor):
    inv_media_vetor = 0
    vetor = np.asarray(vetor)
    inv_media_vetor = (2*np.mean(vetor))-(vetor)
    return inv_media_vetor

def qinv_media(n):
    '''
    :param n: número de qubits
    :return: circuit inversão sobre a média
    '''
    circuito = qiskit.QuantumCircuit(n)

    for qubit in range(n):
        circuito.h(qubit)

    for qubit in range(n):
        circuito.x(qubit)

    circuito.h(n-1)
    circuito.mct(list(range(n-1)), n-1)  
    circuito.h(n-1)

    for qubit in range(n):
        circuito.x(qubit)

    for qubit in range(n):
        circuito.h(qubit)

    return circuito


def oraculo_trivial(n, k=0):

    diagonal_elements = np.ones(2**n)
    diagonal_elements[k] = diagonal_elements[k]*-1

    oracle_gate = Diagonal(diagonal_elements)
    oracle_gate.name = "Oracle"

    #---Circuito----#

    circuito_oraculo = qiskit.QuantumCircuit(n)
    circuito_oraculo.append(oracle_gate, list(range(n)))

    return circuito_oraculo

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc

def grover(oraculo):
    
    n = 4
    range_=list(range(n))

    grover_circuit = qiskit.QuantumCircuit(n)

    grover_circuit = initialize_s(grover_circuit,range_)

    grover_circuit.append(oraculo_trivial(n,k=2), range_)
    #grover_circuit.append(oracle_ex3, [0,1,2])

    grover_circuit.append(qinv_media(n), range_)
    grover_circuit.measure_all()
    grover_circuit.draw()

    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(grover_circuit, backend=backend, shots=1024).result()
    answer = results.get_counts()
    print(answer)
    plot_histogram(answer)

    return grover_circuit
