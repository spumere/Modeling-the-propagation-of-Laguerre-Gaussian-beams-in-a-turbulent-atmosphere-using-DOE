import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
from math import factorial
from tqdm import tqdm
import aotools
import sys

class LG_beam:
    # Класс, который будет хранить характеристики пучка Лагерра-Гаусса
    def __init__(self, N, L, z, wavelength, w_0, m, l, A):
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
        # Массив комплексных амплитуд пучка
        self.A = A

class polar_grid:
    # Класс, который будет хранить полярную сетку, необходимую для вычисления
    # комплексной амплитуды пучка
    def __init__(self, N, L):
        # Создание координатной сетки в декартовых координатах
        x = np.linspace(- L/2, L/2, N, endpoint=False)
        y = np.linspace(- L/2, L/2, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Преобразование в полярные координаты
        self.r = np.sqrt(X**2 + Y**2)
        self.phi = np.arctan2(Y, X)  # от -π до π

class DOE:
    # Класс, который будет хранить параметры дифракционного оптического элемента, необходимые
    # для расчета с помощью итерационной процедуры
     def __init__(self, N, L, f, T):
        # Размер двумерного массива, задающего ДОЭ
        self.N = N
        # Фокусное расстояние
        self.f = f
        self.L = L
        # Функция комплексного пропускания ДОЭ
        self.T = T
        # Маска ДОЭ, необходимая для вычисления коэффициента C в алгоритме
        self.mask = np.ones((N, N), dtype=bool)

class turbulence:
    # Класс, который будет хранить параметры турбулентности, необходимые для моделирования
    # фазового экрана
    def __init__(self, N, L, r_0, L_0, l_0, d, Ltr):
        self.N = N
        self.L = L
        self.dx = L / N
        # Параметр Фрида
        self.r_0 = r_0
        # Внешний масштаб турбулентности
        self.L_0 = L_0
        # Внутренний масштаб турбулентности
        self.l_0 = l_0
        # Длина свободного пространства - элементарных слоев 1 и 3
        self.d = d
        # Длина трассы
        self.Ltr = Ltr


def laguerre_gaussian_beam(polar_grid, beam, normalize_to_source = None):
    # Аргумент многочлена Лагерра
    rho_w = 2 * (polar_grid.r**2) / (beam.W_0**2)
    # Модуль топологического заряда
    l = np.abs(beam.l)
    # Многочлен Лагерра L_m^l
    L_poly = eval_genlaguerre(beam.m, abs(beam.l), rho_w)
    # Нормировочная константа
    A_lm = np.sqrt(2*factorial(beam.m)/(np.pi*factorial(l+beam.m)))
    # Комплексная амплитуда пучка Лагерра-Гаусса (z=0)
    E_lg = A_lm * (polar_grid.r/beam.W_0)**l * L_poly * np.exp(-1j*beam.l*polar_grid.phi -
                                                            polar_grid.r**2/beam.W_0**2) 
    # Возвращаем массив комплексных чисел, соответствующих комплексной амплитуде пучка в 
    # точках координатной сетки
    if normalize_to_source is not None:
        # Нормализуем целевой пучок к энергии источника
        energy_source = np.sum(np.abs(normalize_to_source)**2)
        energy_target = np.sum(np.abs(E_lg)**2)
        factor = np.sqrt(energy_source / energy_target)
        E_lg = E_lg * factor

    return E_lg

def fresnel_transform(A, DOE, wavelength, z, inverse = False):
    dx = DOE.L/DOE.N 
    nu_x = np.fft.fftfreq(DOE.N, dx)
    nu_y = np.fft.fftfreq(DOE.N, dx)
    Nu_x, Nu_y = np.meshgrid(nu_x, nu_y, indexing='ij')
    k = 2 * np.pi / wavelength
    H = np.exp(-1j * k * z) * np.exp(1j * np.pi * wavelength * z * (Nu_x**2 + Nu_y**2))
    if inverse == True:
        H = np.conj(H)
    return np.fft.ifft2(np.fft.fft2(A) * H)

def create_areas(N, L_size, R1, R2):
    # Функция для создания областей L1 и L2 в фокальной плоскости
    x = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    y = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    L1 = r <= R1
    L2 = r <= R2
    return L1, L2

def P1(source, current, DOE, L2, C):
    result = fresnel_transform(C*np.abs(source.A)*np.exp(1j*np.angle(current)), DOE, 
                               source.wavelength, DOE.f)
    result[~L2] = 0
    return result

def P2(target, current, L1, L2):
    result = np.zeros_like(current)
    # В области L: берем амплитуду и фазу целевого поля
    result[L1] = target[L1]
    # В области L'/L: берем текущее поле
    L1_setminus_L2 = L2 & (~L1)
    result[L1_setminus_L2] = current[L1_setminus_L2]
    # Вне L': уже 0
    return result

def calculate_C(current, source, mask):
    # Энергия прошедшего пучка W в области D  
    energy_W_in_D = np.sum(np.abs(current[mask])**2)
    # Энергия падающего пучка A в области D  
    energy_A_in_D = np.sum(np.abs(source[mask])**2)
    C = np.sqrt(energy_W_in_D / energy_A_in_D)
    return C

def AAM(source, target, DOE, L1, L2, precision):

    # Шаг 0: задаем начальное приближение
    w = np.zeros_like(target.A)
    w[L1] = target.A[L1]
    L1_setminus_L2 = L2 & (~L1)
    mu = np.max(np.abs(target.A[L2]))
    num_random_points = np.sum(L1_setminus_L2)
    random_amplitude = np.random.uniform(0, mu, num_random_points)
    random_phase = np.random.uniform(-np.pi, np.pi, num_random_points)
    random_field = random_amplitude * np.exp(1j * random_phase)
    w[L1_setminus_L2] = random_field

    errors = []
    alpha1 = 1
    alpha2 = 1

    for i in tqdm(range(sys.maxsize), desc="Итерации алгоритма"):
        # Шаг 1: применяем Т_2 в плоскости изображения
        w_n = w.copy()
        P2w = P2(target.A, w, L1, L2)
        T2w = w + alpha2 * (P2w - w)
        d1 = np.sqrt(np.sum(np.abs(w - w_n)**2))

        # Шаг 2: обратное преобразование Френеля
        W = fresnel_transform(T2w, DOE, source.wavelength, DOE.f, inverse = True)

        # Шаг 3: применяем Т1 в плоскости ДОЭ
        w_n = w.copy()
        C = calculate_C(W, source.A, DOE.mask)
        P1w = P1(source, W, DOE, L2, C)
        w = T2w + alpha1 * (P1w  - T2w)
        d2 = np.sqrt(np.sum(np.abs(w - w_n)**2))
        # Шаг 4: анализируем ошибку
        error = d1 + d2
        errors.append(error)
        if error < precision:
            break
        if i > 2 and abs(errors[i] - errors[i-1]) < precision:
            break
    # Шаг 5: рассчитываем функцию комплексного пропускания ДОЭ
    T =  np.exp(1j*np.angle(fresnel_transform(w, DOE, source.wavelength, DOE.f, 
                                              inverse = True)/source.A))
    return T, errors

def DOE_propagation(source, DOE):
    # Функция для моделирования распространения пучка источника на фокусное расстояние
    # после прохождения ДОЭ
    return fresnel_transform(source.A * np.exp(1j * np.angle(DOE.T)), DOE,
                             source.wavelength, DOE.f)

def energy_efficiency(result, source, L1):
    numerator = np.sum(np.abs(result[L1])**2)
    denominator = np.sum(np.abs(source)**2)
    print("Энергетическая эффективность: ", numerator/denominator * 100, "%")
    
def correlation(result, target, L1):
    corr_L1 = np.abs(np.sum(result[L1] * np.conj(target[L1])))**2 / (
    np.sum(np.abs(result[L1])**2) * np.sum(np.abs(target[L1])**2))
    print(f"Содержание заданной моды в L1: {corr_L1}")

def mask(field, L1):
    # Создание маски для визуализации
    masked = np.zeros_like(field)
    masked[L1] = field[L1]
    return masked

def one_layer_propagation(w, DOE, turbulence_1, wavelength):
    # Формирование фазового экрана
    phase = aotools.turbulence.phasescreen.ft_phase_screen(turbulence_1.r_0, turbulence_1.N, 
                                                           turbulence_1.dx, turbulence_1.L_0, 
                                                           turbulence_1.l_0)
    S = np.exp(1j * phase)
    step = turbulence_1.d/2
    # Шаг 1
    w = fresnel_transform(w, DOE, wavelength, step)
    # Шаг 2
    W = w * S
    # Шаг 3
    w = fresnel_transform(W, DOE, wavelength, step)
    return w

def propagation(E, DOE, turbulence_1, wavelength):
    Nlayers = int(turbulence_1.Ltr/turbulence_1.d)
    for i in tqdm(range(Nlayers), desc="Моделирование слоев"):
        E = one_layer_propagation(E, DOE, turbulence_1, wavelength)
    return E

def plot_propagation(L, result_before, result_after):
    fig, ax = plt.subplots(2, 2, figsize=(4, 4))
    extent_val = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]
    phase_before = np.angle(result_before)
    im1 = ax[0, 0].imshow(phase_before, cmap='gray', aspect='equal', 
                          extent=extent_val)
    ax[0, 0].axis('off')
    amp_before = np.abs(result_before)
    im2 = ax[0, 1].imshow(amp_before, cmap='gray', aspect='equal', 
                          extent=extent_val)
    ax[0, 1].axis('off')
    phase_after = np.angle(result_after)
    im3 = ax[1, 0].imshow(phase_after, cmap='gray', aspect='equal', 
                          extent=extent_val)
    ax[1, 0].axis('off')
    amp_after = np.abs(result_after)
    im4 = ax[1, 1].imshow(amp_after, cmap='gray', aspect='equal', 
                          extent=extent_val)
    ax[1, 1].axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, 
                       wspace=0.05, hspace=0.05)
    
    plt.show()

# Общие параметры
# Число отсчетов
N = 1024
# Размер области, приближаемой двумерным массивом
L = 0.005
# Длина волны пучка
wavelength = 1550e-9 
# Радиус перетяжки
W_0 = 0.001
p_grid = polar_grid(N, L)

# Параметры источника и создание массива, задающего пучок
source = LG_beam(N, L, 0, wavelength, W_0, 0, 0, 0)
source.A = laguerre_gaussian_beam(p_grid, source)

# Параметры целевого пучка и создание массива, задающего пучок
# Абсолютное значение топологического заряда и радиальный индекс
l, m = 2, 0
target = LG_beam(N, L, 0, wavelength, W_0, m, l, 0)
target.A = laguerre_gaussian_beam(p_grid, target, source.A)

# Параметры ДОЭ и создание экземпляра класса, задающего ДОЭ
f = 0.05
phase_plate = DOE(N, L, f, 0)

# Параметры алгоритма и создание областей
# Допустимая ошибка суммарного расстояния
precision = 5e-4
# Радиусы основной и дополнительной областей
R1 = 0.0015
R2 = 0.0022
L1, L2 = create_areas(N, L, R1, R2)

# Параметры турбулентности
d = 100
Ltr = 1000
r_0 = 0.02
L_0 = 100
l_0 = 0.001
# Создание экземпляра класса, содержащего параметры турбулентности
turbulence_1 = turbulence(N, L, r_0, L_0, l_0, d, Ltr)

# Рассчитываем ДОЭ и результат его применения к исходному пучку
phase_plate.T, errors = AAM(source, target, phase_plate, L1, L2, precision)
result = DOE_propagation(source, phase_plate)

# Рассчитываем характеристики качества полученной моды
correlation(result, target.A, L1)
energy_efficiency(result, source.A, L1)

result_after_turbulence = propagation(result, phase_plate, turbulence_1, wavelength)
# Выводим тепловые карты фазового и амплитудного профиля полученного пучка до и после 
# распространения в атмосфере
plot_propagation(L, result, result_after_turbulence)

# Рассчитываем характеристики качества распространенной моды
correlation(result_after_turbulence, target.A, L1)
energy_efficiency(result_after_turbulence, target.A, L1)
