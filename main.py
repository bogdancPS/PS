import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import math
import time
import scipy.io.wavfile
def f(t):
    return np.sin(2*np.pi*t*3) + np.sin(2*np.pi*t*6) + np.cos(2*np.pi*t*13)


def dft(x, n):

    v = np.linspace(0, 1, n)
    F = np.empty((x, n), dtype=complex)

    for o in range(0, x):
        F[o] = math.e**(-2j*np.pi*o*v)

    m = np.dot(F, f(v))
    M = []
    for i in m:
        M.append(np.abs(i))

    #plt.stem(range(0, x), np.abs(M))
    #plt.show()

    return M


def e1():
    N = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    dftV = []
    fftV = []

    for i in N:
        start = time.process_time() #process_time ne indica timpul codului pe procesor, nu timpul real in care calculatorul executa si alte lucruri
        dft(int(np.floor(i/2)), i)
        end = time.process_time()
        dftV.append(end - start)

        v = np.linspace(0, 1, i)
        start = time.process_time()
        np.fft.fft(f(v))
        end = time.process_time()
        fftV.append(end - start)

    fftVs = [item + 10 for item in fftV]
    dftVs = [item + 10 for item in dftV]
    plt.plot(N, dftVs)
    plt.plot(N, fftVs)
    #plt.yscale('log')
    #plt.scatter(N, np.zeros(len(N)) + 10)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

# start = time.process_time()
# dft(80, 8000)
# end = time.process_time()
# print(end - start)
#
# v = np.linspace(0, 1, 100)
# v = f(v)
# x = np.fft.fft(v)
# X = []
# for i in x:
#     X.append(np.abs(i))
# plt.stem(np.linspace(0, 1, len(X)) * 100, X)
# plt.show()



def e2():
    def f1(t):
        return np.cos(2*np.pi*t*8)

    def f2(t):
        return np.cos(2*np.pi*t*2)

    def f3(t):
        return np.cos(2*np.pi*t*-1)

    v = np.linspace(0, 1, 1000)
    e = np.linspace(0 ,1, 4) # am luat frecventa de esantionare 3, si atunci putem determina in mod sigur doar semnale de freq 1.5 sau mai putin
                                            # am gasit ca doua semnale respecta ce ni se cere daca diferenta freq lor este un multiplu de freq de esantionare (3 in cazul nostru)

    plt.plot(v, f1(v), color='green')
    plt.stem(e, f1(e))
    plt.plot(v, f2(v))
    plt.plot(v, f3(v))
    plt.show()


def e3():
    def f1(t):
        return np.cos(2*np.pi*t*8)

    def f2(t):
        return np.cos(2*np.pi*t*25)

    def f3(t):
        return np.cos(2*np.pi*t*-9)

    v = np.linspace(0, 1, 1000)
    e = np.linspace(0, 1, 18) #freq the esantionare este 17, mai mare decat 2*8
    #ca f1 si f2 sa pice in aceleasi valori in esantionele noastre, trebuie ca diferenta freq lor sa fie multiplu de 17
    #cum f1 are freq 8, daca scadem 17, obtinem -9, un semnal cu freventa 9, mai mare decat 8
    #daca adunam 17 obtinem un semnal de freq 25, mai mare iar decat 8
    #cu alte cuvinte, orice semnal care va pica pe esantionele noastre in aceleasi valori cu f1, va avea cu siguranta o freq mai mare

    plt.plot(v, f1(v))
    plt.stem(e, f1(e))
    plt.plot(v, f2(v))
    plt.plot(v, f3(v))
    plt.show()

def e4():
    print("400 (+epsilon), adica dublu celei mai mari frecvente pe care o poate produce contrabasul")


def e6():
    rate, x = scipy.io.wavfile.read('vocale.wav')
    #sd.play(x, rate)
    #sd.wait()

    print(len(x))
    u = int(np.floor(len(x) / 100))
    print(int(np.floor(len(x)/100)))


    F = []
    for i in range(0, 195):
        f = []
        a = np.fft.fft(x[int(np.floor(i/2*u)):int(np.floor((i/2+1)*u))])
        for j in a:
            f.append(np.abs(j))
        F.append(f)
        print(F)



#e1()
#e2()
#e3()

e6()



