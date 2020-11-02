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
    #Operador [[1,0],[0,e**(1j*alpha)]] -> U3=0,0,alpha
    circuito.u3(theta=0, phi=0, lam=alpha, qubit=0)
    circuito.draw()
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
    
    circuito.h(alvo)
    circuito.cx(controles[1], alvo)
    circuito.tdg(alvo)
    circuito.cx(controles[0], alvo)
    circuito.t(alvo)
    circuito.cx(controles[1], alvo)
    circuito.tdg(alvo)
    circuito.cx(controles[0], alvo)
    circuito.t(controles[1])
    circuito.t(alvo)
    circuito.h(alvo)
    circuito.cx(controles[0], controles[1])
    circuito.t(controles[0])
    circuito.tdg(controles[1])
    circuito.cx(controles[0], controles[1])

    return circuito

def inicializa_3qubits(vetor_dimensao8):
    '''
    Lab2 - questão 4
    #https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html - > ry, cry, mcry
    '''
    arrays = np.split(vetor_dimensao8, 4)

    #Arvore de Estados
    aux_l = np.empty((4))

    i=0
    for array in arrays:
        aux_l[i] = np.linalg.norm(array)
        i+= 1

    i=0
    s_aux_l = np.empty((2))
    s_arrays = np.split(aux_l, 2)
    for s_array in s_arrays: 
        s_aux_l[i] = np.linalg.norm(s_array)
        i+=1

    t_aux_l = np.linalg.norm(s_aux_l)

    #Definindo os alphas
    
    alpha0 = 2*np.arcsin((s_aux_l[1])/np.sqrt(t_aux_l))

    alpha1 = 2*np.arcsin((aux_l[1])/(s_aux_l[0]))
    alpha2 = 2*np.arcsin((aux_l[3])/(s_aux_l[1]))

    alpha3 = 2*np.arcsin((vetor_dimensao8[1])/(aux_l[0]))
    alpha4 = 2*np.arcsin((vetor_dimensao8[3])/(aux_l[1]))
    alpha5 = 2*np.arcsin((vetor_dimensao8[5])/(aux_l[2]))
    alpha6 = 2*np.arcsin((vetor_dimensao8[7])/(aux_l[3]))
    print("Alpha6: ", alpha6)

    #Circuito
    
    qr = qiskit.QuantumRegister(3, 'q')
    circuito = qiskit.QuantumCircuit(qr)
    
    # ------------------------
    # Seu código aqui
    # ------------------------

    circuito.ry(alpha0,qr[0])
    
    circuito.x(qr[0])
    circuito.cry(alpha1,qr[0],qr[1]) #if bit[0] = 0 faz o controle no bit[1]
    
    circuito.x(qr[0])
    circuito.cry(alpha2,qr[0],qr[1]) #if bit[0] = 1 faz o controle no bit[1]
    
    circuito.x(qr[1])
    circuito.x(qr[0])
    circuito.mcry(theta=alpha3, q_controls=qr[0:2], q_target=qr[2], mode='noancilla',q_ancillae=None) #multi control bit; if bit[0] e bit[1] ambos são 0, entao faz o ry no bit 2
    
    circuito.x(qr[1])
    circuito.mcry(theta=alpha4, q_controls=qr[0:2], q_target=qr[2], mode='noancilla',q_ancillae=None) #multi control bit; if bit[0]=0 e bit[1]=1 entao faz o ry no bit 2

    circuito.x(qr[0])
    circuito.x(qr[1])
    circuito.mcry(theta=alpha5, q_controls=qr[0:2], q_target=qr[2], mode='noancilla',q_ancillae=None) #multi control bit; if bit[0]=1 e bit[1]=0 entao faz o ry no bit 2

    circuito.x(qr[1])
    circuito.mcry(theta=alpha6, q_controls=qr[0:2], q_target=qr[2], mode='noancilla',q_ancillae=None) #multi control bit; if bit[0]=1 e bit[1]=1 entao faz o ry no bit 2

    return circuito

def inicializa(vetor):
    '''
    Lab2 - questão 5 - opcional
    '''
    circuito = qiskit.QuantumCircuit()

    return circuito
<<<<<<< HEAD

v8D = np.array([np.sqrt(0.03),np.sqrt(0.07),np.sqrt(0.15),np.sqrt(0.05),np.sqrt(0.1),np.sqrt(0.3),np.sqrt(0.2),np.sqrt(0.1)])
circuito = inicializa_3qubits(v8D)
print(circuito.draw())
=======
>>>>>>> be5075d2ab00647814d35b926fe355857ef57097
