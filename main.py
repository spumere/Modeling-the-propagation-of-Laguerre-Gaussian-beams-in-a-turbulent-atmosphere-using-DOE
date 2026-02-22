import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
from math import factorial
from tqdm import tqdm
import aotools
import sys


class LG_beam:
    # Класс, который будет хранить характеристики пучка Лагерра-Гаусса
    def __init__(self, N, L, z, wavelength, w_0, n, m, A):
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
        self.n = n
        # Абсолютное значение топологического заряда
        self.m = m
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

def laguerre_gaussian_beam(polar_grid, beam):
    # Аргумент многочлена Лагерра
    rho_w = 2 * (polar_grid.r**2) / (beam.W_0**2)
    # Многочлен Лагерра L_m^l
    L = eval_genlaguerre(beam.n, abs(beam.m), rho_w)
    # Нормировочная константа
    A_lm = np.sqrt(2*factorial(beam.n)/(np.pi*factorial(beam.n+beam.m)))
    # Комплексная амплитуда пучка Лагерра-Гаусса (z=0)
    E_lg = 1/beam.W_0 * A_lm * (np.sqrt(2)*polar_grid.r/beam.W_0)**beam.m * L *np.exp(
        1j*beam.m*polar_grid.phi - polar_grid.r**2/beam.W_0**2) 
    # Возвращаем массив комплексных чисел, соответствующих комплексной амплитуде пучка в 
    # точках координатной сетки
    return E_lg

def ASM(A, DOE, wavelength, z, inverse=False):
    dx = DOE.L/DOE.N 
    nu_x = np.fft.fftfreq(DOE.N, dx)
    nu_y = np.fft.fftfreq(DOE.N, dx)
    Nu_x, Nu_y = np.meshgrid(nu_x, nu_y, indexing='ij')
    
    sqrt_argument = wavelength**(-2) - Nu_x**2 - Nu_y**2
    
    H = np.zeros_like(sqrt_argument, dtype=complex)
    
    # Распространяющиеся волны
    propagating_mask = sqrt_argument >= 0
    H[propagating_mask] = np.exp(-1j * 2 * np.pi * z * np.sqrt(sqrt_argument[propagating_mask]))
    
    # Эванесцентные волны 
    evanescent_mask = sqrt_argument < 0
    H[evanescent_mask] = 0
    
    if inverse:
        # Для обратного распространения
        H[propagating_mask] = np.exp(1j * 2 * np.pi * z * np.sqrt(sqrt_argument[propagating_mask]))
        # Для обратных эванесцентных волн
        H[evanescent_mask] = 0
    
    return np.fft.ifft2(np.fft.fft2(A) * H)

def create_areas(N, L_size, R1, R2):
    # Функция для создания полезной L1 и дополнительной L\L1 = L2
    # областей в фокальной плоскости
    x = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    y = np.linspace(-L_size/2, L_size/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    L1 = r <= R1
    L2 = r <= R2
    return L1, L2

def P1(source, current, DOE, L2):
    result = ASM(np.abs(source.A)*np.exp(1j*np.angle(current)), DOE, 
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
        # Шаг 1: применяем Т2 в плоскости изображения
        w_n = w.copy()
        P2w = P2(target.A, w, L1, L2)
        T2w = w + alpha2 * (P2w - w)
        d1 = np.sqrt(np.sum(np.abs(P2w - w_n)**2))

        # Шаг 2: обратное преобразование Френеля
        W = ASM(T2w, DOE, source.wavelength, DOE.f, inverse = True)

        # Шаг 3: применяем Т1 в плоскости ДОЭ
        w_n = w.copy()
        P1w = P1(source, W, DOE, L2)
        w = T2w + alpha1 * (P1w  - T2w)
        d2 = np.sqrt(np.sum(np.abs(P1w - w_n)**2))
        # Шаг 4: анализируем ошибку
        error = d1 + d2
        errors.append(error)
        if error < precision:
            break
        if i > 2 and abs(errors[i] - errors[i-1]) < precision:
            break
    # Шаг 5: рассчитываем функцию комплексного пропускания ДОЭ
    T =  np.exp(1j*np.angle(ASM(w, DOE, source.wavelength, DOE.f, 
                                              inverse = True)/source.A))
    return T

def DOE_propagation(source, DOE, L1):
    # Функция для моделирования распространения пучка источника на фокусное расстояние
    # после прохождения ДОЭ
    return mask(ASM(source.A * np.exp(1j * np.angle(DOE.T)), DOE,
                             source.wavelength, DOE.f), L1)
    
def correlation(result, target, L1):
    return np.abs(np.sum(result[L1] * np.conj(target[L1])))**2 / (
    np.sum(np.abs(result[L1])**2) * np.sum(np.abs(target[L1])**2))

def mask(field, L1):
    # Создание маски для визуализации
    masked = np.zeros_like(field)
    masked[L1] = field[L1]
    return masked

def make_turbulence(N, turbulence):
    phase_tensor = np.empty((N, turbulence.N, turbulence.N), dtype=np.float32)
    for i in range(N):
        phase = aotools.turbulence.phasescreen.ft_phase_screen(turbulence.r_0, turbulence.N, 
                                                           turbulence.dx, turbulence.L_0, 
                                                           turbulence.l_0)
        phase_tensor[i] = phase
    return phase_tensor

def one_layer_propagation(w, DOE, turbulence_1, wavelength, phase):
    # Формирование фазового экрана
    S = np.exp(1j * phase)
    step = turbulence_1.d/2
    # Шаг 1
    w = ASM(w, DOE, wavelength, step)
    # Шаг 2
    W = w * S
    # Шаг 3
    w = ASM(W, DOE, wavelength, step)
    return w

def propagation(E, DOE, turbulence_1, wavelength, phase_tensor):
    Nlayers = int(turbulence_1.Ltr/turbulence_1.d)
    for i in tqdm(range(Nlayers), desc="Моделирование слоев"):
        E = one_layer_propagation(E, DOE, turbulence_1, wavelength, phase_tensor[i])
    return E

def plot_propagation(L, all_before, all_after_1, all_after_2, save_svg=False, filename='all_modes.svg'):
    fig = plt.figure(figsize=(10, 8))
    
    extent_val = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]
    mode_names = ['Гауссов пучок', 'ГЛ (0,1)', 'ГЛ (0,2)', 'ГЛ (0,3)', 'ГЛ (0,4)', 'ГЛ (0,5)']
    
    n_rows, n_cols = 6, 6
    
    left_margin = 0.10   
    right_margin = 0.04 
    bottom_margin = 0.06 
    top_margin = 0.08  
    
    grid_width = 1.0 - left_margin - right_margin
    grid_height = 1.0 - bottom_margin - top_margin
    
    frame_width = grid_width / n_cols
    frame_height = grid_height / n_rows
    
    grid_start_x = left_margin
    grid_start_y = bottom_margin
    
    im_for_cbar = {}

    for row in range(n_rows):
        for col in range(n_cols):

            x0 = grid_start_x + col * frame_width
            y0 = grid_start_y + row * frame_height
            
            ax = fig.add_axes([x0, y0, frame_width, frame_height])
            
            if row == 5:
                phase = np.angle(all_before[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_before'] = (im, x0 + frame_width, y0)
            
            elif row == 4:
                amp = np.abs(all_before[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            elif row == 3:
                phase = np.angle(all_after_1[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_r1'] = (im, x0 + frame_width, y0)
            
            elif row == 2:  
                amp = np.abs(all_after_1[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            elif row == 1: 
                phase = np.angle(all_after_2[col])
                im = ax.imshow(phase, cmap='gray', aspect='equal',
                              extent=extent_val, vmin=-np.pi, vmax=np.pi)
                if col == n_cols - 1:
                    im_for_cbar['phase_r2'] = (im, x0 + frame_width, y0)
            
            elif row == 0: 
                amp = np.abs(all_after_2[col])
                ax.imshow(amp, cmap='gray_r', aspect='equal', extent=extent_val)
            
            ax.axis('off')
    

    for key, (im, x_pos, y_pos) in im_for_cbar.items():
        cbar_width = 0.01  
        cbar_x = x_pos + 0.005  
        cbar_ax = fig.add_axes([cbar_x, y_pos, cbar_width, frame_height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_ticks([-np.pi + 0.2, 0, np.pi - 0.2])
        cbar.set_ticklabels(['-π', '0', 'π'])
        cbar.ax.tick_params(labelsize=6) 
    
    row_labels = ['Ампл. \n(r0=0.02)', 'Фаза \n(r0=0.02)',
                  'Ампл. \n(r0=0.01)', 'Фаза \n(r0=0.01)',
                  'Ампл. до', 'Фаза до']
    
    for row in range(n_rows):
        text_x = left_margin - 0.008  
        text_y = grid_start_y + row * frame_height + frame_height/2
        fig.text(text_x, text_y, row_labels[row], 
                fontsize=10, va='center', ha='right', 
                linespacing=1.2)  
    
    for col in range(n_cols):
        text_x = grid_start_x + col * frame_width + frame_width/2
        text_y = grid_start_y + n_rows * frame_height + 0.005 
        fig.text(text_x, text_y, mode_names[col], 
                fontsize=10, va='bottom', ha='center')  
    
    if save_svg:
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"График сохранен как {filename}")
    
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
source.A = laguerre_gaussian_beam(p_grid, source) / np.sqrt(np.sum(np.abs(
    laguerre_gaussian_beam(p_grid, source))**2))
# Параметры целевого пучка и создание массива, задающего пучок
# Абсолютное значение топологического заряда и радиальный индекс
m, n = 1, 0
target = LG_beam(N, L, 0, wavelength, W_0, n, m, 0)
target.A = laguerre_gaussian_beam(p_grid, target) / np.sqrt(np.sum(np.abs(
    laguerre_gaussian_beam(p_grid, target))**2))

# Параметры ДОЭ и создание экземпляра класса, задающего ДОЭ
f = 0.05
phase_plate = DOE(N, L, f, 0)

# Параметры алгоритма и создание областей
# Допустимая ошибка суммарного расстояния
precision = 1e-4
# Радиусы основной и дополнительной областей
R1 = 0.0015
R2 = 0.0022
L1, L2 = create_areas(N, L, R1, R2)
# Параметры турбулентности
# Расстояние между фазовыми экранами
d = 100
# Длина трека
Ltr = 1000
# Параметр Фрида
r_0_1 = 0.01
r_0_2 = 0.02
# Внешний и внутренний масштабы турбулентности
L_0 = 20
l_0 = 0.001
# Создание экземпляра класса, содержащего параметры турбулентности
turbulence_1 = turbulence(N, L, r_0_1, L_0, l_0, d, Ltr)
turbulence_2 = turbulence(N, L, r_0_2, L_0, l_0, d, Ltr)

# Создание трехмерных массивов, содержащих все фазовые экраны для трека
phase_tensor_1 = make_turbulence(int(Ltr/d), turbulence_1)
phase_tensor_2 = make_turbulence(int(Ltr/d), turbulence_2)
# Список для хранения массивов пучков до прохождения через трек
all_results_before = []
# Список для хранения массивов пучков, прошедших через трек с r0 = 1 см
all_results_after_1 = []
# Список для хранения массивов пучков, прошедших через трек с r0 = 2 см
all_results_after_2 = []

print("Гаусс")
result_after_turbulence_1 = propagation(source.A, phase_plate, turbulence_1, 
                                        wavelength, phase_tensor_1)
result_after_turbulence_2 = propagation(source.A, phase_plate, turbulence_2, 
                                        wavelength, phase_tensor_2)
all_results_before.append(source.A)
all_results_after_1.append(mask(result_after_turbulence_1, L1))
all_results_after_2.append(mask(result_after_turbulence_2, L1))
print(correlation(result_after_turbulence_1, source.A, L1))
correlation(result_after_turbulence_2, source.A, L1)
np.save('turbulence.npy', phase_tensor_1)
# Создаем массивы для всех мод
modes = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

for i, (m, n) in enumerate(modes):
    if m == 0 and n == 0:
        # Гауссов пучок уже обработан
        continue
    
    print(f"ГЛ{m:02d}")
    target = LG_beam(N, L, 0, wavelength, W_0, n, m, 0)
    target.A = laguerre_gaussian_beam(p_grid, target) / np.sqrt(np.sum(np.abs(
        laguerre_gaussian_beam(p_grid, target))**2))
    
    phase_plate.T = AAM(source, target, phase_plate, L1, L2, precision)
    result = DOE_propagation(source, phase_plate, L1)
    correlation(result, target.A, L1)
    result_after_turbulence_1 = propagation(result, phase_plate, turbulence_1, wavelength, 
                                            phase_tensor_1)
    result_after_turbulence_2 = propagation(result, phase_plate, turbulence_2, wavelength,
                                            phase_tensor_2)
    correlation(result_after_turbulence_1, target.A, L1)
    correlation(result_after_turbulence_2, target.A, L1)
    all_results_before.append(result)
    all_results_after_1.append(mask(result_after_turbulence_1, L1))
    all_results_after_2.append(mask(result_after_turbulence_2, L1))

# Теперь преобразуем списки в 3D массивы
results_before_3d = np.stack(all_results_before)
results_after_1_3d = np.stack(all_results_after_1)
results_after_2_3d = np.stack(all_results_after_2)

# И отображаем все сразу
plot_propagation(L, results_before_3d, results_after_1_3d, results_after_2_3d,
                 save_svg=True, filename='all_modes.svg')
