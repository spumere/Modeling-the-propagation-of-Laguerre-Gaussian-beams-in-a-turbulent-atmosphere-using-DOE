import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
from math import factorial
import aotools
from tqdm import tqdm

class LG_beam:
    # Класс, который будет характеристики пучка Лагерра-Гаусса
    def __init__(self, N, L, z, wavelength, w_0, m, l):
        # Размер двумерного массива, задающего пучок
        self.N = N
        # Размер области, приближаемой массивом точек
        self.L = L
        # Координата пучка по оси z
        self.z = z
        # Длина волны пучка
        self.wavelength = wavelength
        # Радиус перетяжки
        self.W_0 = w_0
        # Радиальный индекс
        self.m = m
        # Абсолютное значение топологического заряда
        self.l = l

class polar_grid:
    # Класс, который будет хранить полярную сетку, необходимую для вычисления
    # комплексной амплитуды пучка
    def __init__(self, N, L):
        # Шаг сетки
        dx = L / N
        # Создание координатной сетки в декартовых координатах
        x = np.linspace(- L/2, L/2, N, endpoint=False)
        y = np.linspace(- L/2, L/2, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Преобразование в полярные координаты
        self.r = np.sqrt(X**2 + Y**2)
        self.phi = np.arctan2(Y, X)  # от -π до π

class frequency_grid:
    # Класс, который будет хранить сетку пространственных частот, необоходимую для вычисления
    # передаточной функции свободного пространства
    def __init__(self, N, L):
        # Шаг сетки
        dx = L / N
        # Генерируем массив пространственных частот дискретного преобразования Фурье
        nu_x = np.fft.fftfreq(N, dx)
        nu_y = np.fft.fftfreq(N, dx)
        # Формируем двумерную сетку пространственных частот
        Nu_x, Nu_y = np.meshgrid(nu_x, nu_y)
        self.nu_x = Nu_x
        self.nu_y = Nu_y

def laguerre_gaussian_beam(polar_grid, beam):
    # Аргумент многочлена Лагерра
    rho_w = 2 * (polar_grid.r**2) / (beam.W_0**2)
    # Многочлен Лагерра L_m^l
    L_poly = eval_genlaguerre(beam.m, abs(beam.l), rho_w)
    # Нормировочная константа
    A_lm = np.sqrt(2*factorial(beam.m)/(np.pi*factorial(beam.l+beam.m)))
    # Комплексная амплитуда пучка Лагерра-Гаусса (z=0)
    E_lg = A_lm * (polar_grid.r/beam.W_0)**beam.l * L_poly * np.exp(-1j*beam.l*polar_grid.phi -
                                                            polar_grid.r**2/W_0**2) 
    # Возвращаем массив комплексных чисел, соответствующих комплексной амплитуде пучка в 
    # точках координатной сетки
    return E_lg


def one_layer_propagation(freq_grid, E, d, wavelength):
    k = 2*np.pi/wavelength
    # Передаточная функция свободного пространства
    H = np.exp(-1j*2*np.pi*d*np.sqrt(wavelength**(-2) - freq_grid.nu_x**2 - freq_grid.nu_y**2))
    # Формирование фазового экрана
    phase = aotools.turbulence.phasescreen.ft_phase_screen(r_0, N, dx, L0, l0)
    S = np.exp(1j * phase)
    # Шаг 1
    F = np.fft.fft2(E)
    F1 = F * H
    E1 = np.fft.ifft2(F1)
    # Шаг 2
    E2 = E1 * S
    # Шаг 3
    F2 = np.fft.fft2(E2)
    F3 = F2 * H
    E3 = np.fft.ifft2(F3)
    return E3

def propagation(freq_grid, E, d, wavelength, Ltr):
    Nlayers = int(Ltr/d)
    for i in tqdm(range(Nlayers), desc="Моделирование слоев"):
        E = one_layer_propagation(freq_grid, E, d, wavelength)
    return E

def phase_visualization(L, E):
    #Визуализируем фазу, поэтому создаем массив аргументов комплексных амплитуд
    phase = np.angle(E)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(phase, cmap='gray', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
    plt.colorbar(label='Фаза (рад)')
    plt.xlabel('x, мм')
    plt.ylabel('y, мм')
    plt.show()

# Параметры пучка
N = 512  
L = 0.02
# Шаг сетки
dx = L / N
# Длина волны пучка
wavelength = 1550e-9 
# Длина свободного пространства - элементарных слоев 1 и 3
d = 100
# Длина трассы
Ltr = 1000
W_0 = 5e-3
m, l = 1, 1
p_grid = polar_grid(N, L)
# Создание сетки пространственных частот
f_grid = frequency_grid(N, L)
# Параметры турбулентности
# Параметр Фрида
r_0 = 0.02
# Внешний масштаб турбулентности
L0 = 100
# Внутренний масштаб турбулентности
l0 = 0.001

beam = LG_beam(N, L, 0, wavelength, W_0, m, l)

E = laguerre_gaussian_beam(p_grid, beam)

E3 = propagation(f_grid, E, d, wavelength, Ltr)
# Выводим тепловую карту фазового профиля пучка до распространения в атмосфере
phase_visualization(L, E)
# Выводим тепловую карту фазового профиля пучка после распространения в атмосфере
phase_visualization(L, E3)
