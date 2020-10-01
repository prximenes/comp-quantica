import qiskit
import numpy as np

def rzryrz(U):
    '''
    Lab2 - questão 1
    :param U: matriz unitária 2 x 2
    :return: [alpha, beta, gamma e delta]
            U = e^(1j * alpha) * Rz(beta) * Ry(gamma) * Rz(delta)

    U = [[(1+1j)/2, (1-1j)/2],
        [(1-1j)/2, (1+1j)/2]]
    '''
    # -----------------
    # Seu código aqui
    x00 = np.angle(U[0][0])
    x01 = np.angle(U[0][1])
    x10 = np.angle(U[1][0])
    x11 = np.angle(U[1][1])

    A = np.array([[1, -0.5, -0.5], [1, -0.5, 0.5], [1, 0.5, -0.5]])
    B = np.array([x00, x01 + np.pi, x10])
    X = np.linalg.inv(A).dot(B)    

    alpha = X[0]
    beta = X[1]
    gamma = np.arccos(np.abs(U[0][0]))*2
    delta = X[2]
    # -----------------

    return [alpha, beta, gamma, delta]

def operador_controlado(V):
    '''
    Lab2 - questão 2
    :param V: matriz unitária 2 x 2
    :return: circuito quântico com dois qubits aplicando o
             o operador V controlado.
    '''
    #Decomponsição de V

    x00 = np.angle(V[0][0])
    x01 = np.angle(V[0][1])
    x10 = np.angle(V[1][0])
    x11 = np.angle(V[1][1])

    A = np.array([[1, -0.5, -0.5], [1, -0.5, 0.5], [1, 0.5, -0.5]])
    B = np.array([x00, x01 + np.pi, x10])
    X = np.linalg.inv(A).dot(B)    

    alpha = X[0]
    beta = X[1]
    gamma = np.arccos(np.abs(V[0][0]))*2
    delta = X[2]

    #-------Matrizes------#
    #A = Rz(beta)*Ry(gamma/2)
    #B = Ry(-1*gamma/2)*Rz(-1*(delta + beta)/2)
    #C = Rz((delta-beta)/2)

    #-------Circuito------#
    circuito = qiskit.QuantumCircuit(2)
    #C
    circuito.rz((delta-beta)/2, 1)
    #cx
    circuito.cx(0,1)
    #B
    circuito.rz(-1*(delta + beta)/2, 1)
    circuito.ry(-1*gamma/2, 1)
    #cx
    circuito.cx(0,1)
    #A
    circuito.ry(gamma/2, 1)
    circuito.rz(beta, 1)
    circuito.draw()
    #Operador [[1,0],[0,e**(1j*alpha)]] -> U3=0,0,alpha
    circuito.u3(theta=0, phi=0, lam=alpha, qubit=0)
    return circuito


def toffoli():
    '''
    Lab2 - questão 3
    :param n: número de controles
    :param V:
    :return: circuito quântico com n+1 qubits + n-1 qubits auxiliares
            que aplica o operador nCV.
    '''
    controles = qiskit.QuantumRegister(2)
    alvo = qiskit.QuantumRegister(1)

    circuito = qiskit.QuantumCircuit(controles, alvo)

    #------------------------
    # Seu código aqui
    # ------------------------
    circuito.cu1(np.pi, controles[1], alvo)
    circuito.cx(controles[0],controles[1])
    circuito.cu1(-np.pi, controles[1], alvo)
    circuito.cx(controles[0],controles[1])
    circuito.cu1(np.pi,controles[0],alvo)
    circuito.draw()
    
    return circuito

def inicializa_3qubits(vetor_dimensao8):
    '''
    Lab2 - questão 4
    '''

    circuito = qiskit.QuantumCircuit(3)

    # ------------------------
    # Seu código aqui
    # ------------------------

    return circuito

def inicializa(vetor):
    '''
    Lab2 - questão 5 - opcional
    '''
    circuito = qiskit.QuantumCircuit()

    return circuito