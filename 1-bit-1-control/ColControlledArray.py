import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.colors import Normalize
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

# 假设 ArrayTools.py 与此脚本在同一目录下
from ArrayTools import ArrayTools

# 尝试导入cupy，如果失败则将其设为None
try:
    import cupy as cp
except ImportError:
    cp = None

class ColControlledArray(ArrayTools):
    """
    一个专用于列控反射阵列的类，继承自ArrayTools。
    它封装了初始相位的全局优化、特定方向的列控密码计算（后处理）
    以及结果可视化等功能。
    能根据硬件环境（是否可用GPU）和问题规模（阵列列数M）自动选择最优求解策略。
    """
    
    # 定义求解列控密码本u的方法切换阈值
    BRUTEFORCE_THRESHOLD = 16

    def __init__(self, n_elements_xy, p, freq, r_feed, hpbw_e, hpbw_h, comp_phase_matrix=None, isdebug=0):
        """
        初始化列控阵列实例。
        """
        # 如果未提供初始相位矩阵，则创建一个默认的全零矩阵
        if comp_phase_matrix is None:
            comp_phase_matrix = np.zeros((n_elements_xy[1], n_elements_xy[0]), dtype=int)
            
        # 调用父类的构造函数
        super().__init__(n_elements_xy, p, freq, r_feed, hpbw_e, hpbw_h, comp_phase_matrix, isdebug)
        
        if self.isdebug:
            print("ColControlledArray 子类已初始化。")
            print(f"列控求解器暴力搜索阈值 M = {self.BRUTEFORCE_THRESHOLD}")

    # --------------------------------------------------------------------------
    # --- 核心求解器方法 (根据M的大小和GPU可用性选择不同策略) ---
    # --------------------------------------------------------------------------

    def find_best_u(self, theta_obj_deg, phi_obj_deg, initial_phi_mn):
        """
        公共方法：根据硬件环境和阵列列数M自动选择最优的求解器来寻找列控密码u。
        """
        if self.use_gpu and self.Nx <= self.BRUTEFORCE_THRESHOLD:
            if self.isdebug: print(f"求解器: CUDA Brute-force (GPU可用, M<={self.BRUTEFORCE_THRESHOLD})")
            return self._find_best_u_bruteforce(initial_phi_mn, theta_obj_deg, phi_obj_deg)
        elif not self.use_gpu:
            if self.isdebug: print(f"求解器: Genetic Algorithm (GPU不可用, CPU多线程)")
            return self._find_best_u_ga_cpu(initial_phi_mn, theta_obj_deg, phi_obj_deg)
        else: # self.use_gpu is True and self.Nx > self.BRUTEFORCE_THRESHOLD
            if self.isdebug: print(f"求解器: Heuristic (GPU可用, M>{self.BRUTEFORCE_THRESHOLD})")
            return self._find_best_u_heuristic(initial_phi_mn, theta_obj_deg, phi_obj_deg)

    def _find_best_u_bruteforce(self, initial_phi_mn, theta_obj_deg, phi_obj_deg):
        """
        私有方法：使用CUDA进行暴力搜索。
        """
        if not self.use_gpu: raise EnvironmentError("暴力搜索需要GPU和cupy。")
        
        xp = self.xp
        M, N, DTYPE = self.Nx, self.Ny, xp.float32

        theta_obj = xp.array(np.radians(theta_obj_deg), dtype=DTYPE)
        phi_obj = xp.array(np.radians(phi_obj_deg), dtype=DTYPE)
        x_1d = (xp.arange(M, dtype=DTYPE) * self.p_mm) - (M - 1) * self.p_mm / 2
        y_1d = (xp.arange(N, dtype=DTYPE) * self.p_mm) - (N - 1) * self.p_mm / 2
        x_grid, y_grid = xp.meshgrid(x_1d, y_1d)

        dist_to_feed = xp.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        hpbw_avg_rad = xp.array(np.radians((self.hpbw_e_deg + self.hpbw_h_deg) / 2), dtype=DTYPE)
        
        try:
            cos_hpbw_half = xp.cos(hpbw_avg_rad / 2)
            if float(cos_hpbw_half) > 0.999999: q = DTYPE(100.0)
            else:
                log_val = xp.log10(cos_hpbw_half)
                q = -3.0 / (20.0 * log_val) if float(log_val) != 0.0 else DTYPE(100.0)
        except (ValueError, ZeroDivisionError): q = DTYPE(25.0)
        q = xp.clip(q, a_min=None, a_max=100.0)
        
        arccos_arg = xp.clip(self.r_feed_mm / dist_to_feed, -1.0, 1.0)
        theta_feed = xp.arccos(arccos_arg)
        feed_pattern_amp = xp.cos(theta_feed)**q
        illumination_amp = (feed_pattern_amp / dist_to_feed).astype(xp.complex64)

        if xp.isinf(illumination_amp).any() or xp.isnan(illumination_amp).any(): return np.zeros(M, dtype=int)

        illumination_phase = (-1 * self.k * dist_to_feed).astype(DTYPE)
        base_term = illumination_amp * xp.exp(1j * illumination_phase)
        initial_phi_rad_gpu = xp.array(initial_phi_mn * np.pi, dtype=DTYPE)
        element_contributions = base_term * xp.exp(1j * initial_phi_rad_gpu)
        space_phase_delay = (self.k * (x_grid * xp.sin(theta_obj) * xp.cos(phi_obj) + y_grid * xp.sin(theta_obj) * xp.sin(phi_obj))).astype(DTYPE)
        
        total_configs = 1 << M
        
        kernel_code = r'''
        #include <cupy/complex.cuh>
        extern "C" __global__
        void compute_power_u_with_steering(const complex<float>* c, const float* s, unsigned int M, unsigned int N, unsigned long long t, float* p) {
            unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= t) return;
            complex<float> E(0.0f, 0.0f);
            for (int m = 0; m < M; m++) {
                float sign = 1.0f - 2.0f * ((idx >> m) & 1);
                complex<float> col_sum(0.0f, 0.0f);
                for (int n = 0; n < N; n++) {
                    int e_idx = n * M + m;
                    col_sum += c[e_idx] * complex<float>(cosf(s[e_idx]), sinf(s[e_idx]));
                }
                E += sign * col_sum;
            }
            p[idx] = E.real() * E.real() + E.imag() * E.imag();
        }
        '''
        power_gpu = xp.zeros(total_configs, dtype=xp.float32)
        compute_power_kernel = xp.RawKernel(kernel_code, 'compute_power_u_with_steering')
        block_size = 256
        grid_size = (total_configs + block_size - 1) // block_size
        compute_power_kernel((grid_size,), (block_size,), (element_contributions, space_phase_delay, xp.uint32(M), xp.uint32(N), xp.uint64(total_configs), power_gpu))
        
        if xp.isinf(power_gpu).any() or xp.isnan(power_gpu).any(): return np.zeros(M, dtype=int)

        best_idx = xp.argmax(power_gpu).get()
        return np.array([(best_idx >> i) & 1 for i in range(M)], dtype=int)

    def _find_best_u_heuristic(self, initial_phi_mn, theta_obj_deg, phi_obj_deg):
        """
        私有方法：使用启发式算法。
        """
        theta_obj_rad, phi_obj_rad = np.radians(theta_obj_deg), np.radians(phi_obj_deg)
        x_1d = np.arange(self.Nx) * self.p_mm - (self.Nx - 1) * self.p_mm / 2
        y_1d = np.arange(self.Ny) * self.p_mm - (self.Ny - 1) * self.p_mm / 2
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)
        dist_to_feed = np.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        illumination_phase = -self.k * dist_to_feed
        space_phase_delay = self.k * (x_grid * np.sin(theta_obj_rad) * np.cos(phi_obj_rad) + y_grid * np.sin(theta_obj_rad) * np.sin(phi_obj_rad))
        ideal_phase = -(illumination_phase + space_phase_delay)
        u_vector, initial_phi_rad = np.zeros(self.Nx, dtype=int), initial_phi_mn * np.pi
        for m in range(self.Nx):
            error_0 = np.sum(np.angle(np.exp(1j * (initial_phi_rad[:, m] - ideal_phase[:, m])))**2)
            error_1 = np.sum(np.angle(np.exp(1j * (initial_phi_rad[:, m] + np.pi - ideal_phase[:, m])))**2)
            if error_1 < error_0: u_vector[m] = 1
        return u_vector

    def _fitness_for_u(self, u_chromosome, initial_phi_mn, theta_obj_deg, phi_obj_deg):
        """
        私有方法：用于求解u向量的遗传算法的适应度函数。
        """
        final_phi = initial_phi_mn.copy()
        for i, flip in enumerate(u_chromosome):
            if flip == 1: final_phi[:, i] = 1 - final_phi[:, i]
        
        final_phase_rad = final_phi * np.pi
        
        theta_obj_rad, phi_obj_rad = np.radians(theta_obj_deg), np.radians(phi_obj_deg)
        x_1d = np.arange(self.Nx) * self.p_mm - (self.Nx - 1) * self.p_mm / 2
        y_1d = np.arange(self.Ny) * self.p_mm - (self.Ny - 1) * self.p_mm / 2
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)
        dist_to_feed = np.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        hpbw_avg_rad = np.radians((self.hpbw_e_deg + self.hpbw_h_deg) / 2)
        q = -3 / (20 * np.log10(np.cos(hpbw_avg_rad / 2)))
        theta_feed = np.arccos(np.clip(self.r_feed_mm / dist_to_feed, -1.0, 1.0))
        feed_pattern_amp = np.cos(theta_feed)**q
        illumination_amp = feed_pattern_amp / dist_to_feed
        illumination_phase = -1 * self.k * dist_to_feed
        far_field_phase_shift = self.k * (x_grid * np.sin(theta_obj_rad) * np.cos(phi_obj_rad) + y_grid * np.sin(theta_obj_rad) * np.sin(phi_obj_rad))
        total_phase = illumination_phase + final_phase_rad + far_field_phase_shift
        E_field_at_target = np.sum(illumination_amp * np.exp(1j * total_phase))
        return np.abs(E_field_at_target)

    def _find_best_u_ga_cpu(self, initial_phi_mn, theta_obj_deg, phi_obj_deg):
        """
        私有方法：当没有GPU时，使用CPU多线程遗传算法寻找u。
        """
        ga_params_u = {'n_generations': 50, 'population_size': 100, 'mutation_rate': 0.05, 'crossover_rate': 0.8, 'elite_size': 5, 'n_jobs': -1}
        
        fitness_func = partial(self._fitness_for_u, initial_phi_mn=initial_phi_mn, theta_obj_deg=theta_obj_deg, phi_obj_deg=phi_obj_deg)
        
        population = np.random.randint(0, 2, size=(ga_params_u['population_size'], self.Nx))
        best_u_overall, best_fitness_overall = population[0], -1

        for generation in range(ga_params_u['n_generations']):
            with Parallel(n_jobs=ga_params_u['n_jobs']) as parallel:
                fitness_scores = parallel(delayed(fitness_func)(u) for u in population)
            fitness_scores = np.array(fitness_scores)
            
            best_idx_gen = np.argmax(fitness_scores)
            if fitness_scores[best_idx_gen] > best_fitness_overall:
                best_fitness_overall = fitness_scores[best_idx_gen]
                best_u_overall = population[best_idx_gen].copy()

            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population = [population[i] for i in sorted_indices[:ga_params_u['elite_size']]]
            
            num_parents_needed = ga_params_u['population_size'] - ga_params_u['elite_size']
            parent_indices = []
            for _ in range(num_parents_needed):
                contenders_indices = np.random.choice(len(population), 5, replace=False)
                winner_index = contenders_indices[np.argmax(fitness_scores[contenders_indices])]
                parent_indices.append(winner_index)
            parents = population[parent_indices]

            offspring = []
            for i in range(0, num_parents_needed, 2):
                p1, p2 = parents[i], parents[i+1] if i + 1 < len(parents) else parents[i]
                if np.random.rand() < ga_params_u['crossover_rate']:
                    crossover_point = np.random.randint(1, self.Nx)
                    c1, c2 = np.concatenate([p1[:crossover_point], p2[crossover_point:]]), np.concatenate([p2[:crossover_point], p1[crossover_point:]])
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
            
            for i in range(len(offspring)):
                mask = np.random.rand(self.Nx) < ga_params_u['mutation_rate']
                offspring[i][mask] = 1 - offspring[i][mask]
            
            new_population.extend(offspring)
            population = np.array(new_population[:ga_params_u['population_size']])
            
        return best_u_overall

    def _fitness_for_phi(self, phi_chromosome, scan_angles_deg, phi_obj_deg, w_deviation, w_sll):
        initial_phi_mn = phi_chromosome.reshape((self.Ny, self.Nx))
        total_deviation, total_sll = 0.0, 0.0
        for theta_obj_deg in scan_angles_deg:
            u_vector = self.find_best_u(theta_obj_deg, phi_obj_deg, initial_phi_mn)
            final_phi_01 = initial_phi_mn.copy()
            for i, flip in enumerate(u_vector):
                if flip == 1: final_phi_01[:, i] = 1 - final_phi_01[:, i]
            
            # MODIFIED: 调用父类中的公有方法
            gain_db, theta_deg = self.calculate_pattern_slice(final_phi_01, phi_obj_deg)
            
            actual_peak_angle = theta_deg[np.argmax(gain_db)]
            total_deviation += abs(actual_peak_angle - theta_obj_deg)
            
            # MODIFIED: 调用父类中的公有方法
            total_sll += (self.get_sidelobe_level(gain_db) + 40)
            
        cost = w_deviation * total_deviation + w_sll * total_sll
        return 1.0 / (cost + 1e-6)

    def optimize_initial_phase(self, ga_params, scan_angles_deg, phi_obj_deg, w_deviation, w_sll):
        def fitness_for_ga(chromosome, sim_instance):
            return self._fitness_for_phi(phi_chromosome=chromosome, scan_angles_deg=scan_angles_deg, phi_obj_deg=phi_obj_deg, w_deviation=w_deviation, w_sll=w_sll)
        initial_population = np.random.randint(0, 2, size=(ga_params['population_size'], self.Ny * self.Nx))
        print("\n--- 开始运行遗传算法优化初始相位 Φ_mn ---")
        best_phi_flat, best_fitness = self.run_genetic_optimization(fitness_function=fitness_for_ga, initial_population=initial_population, ga_params=ga_params)
        if best_phi_flat is not None: return best_phi_flat.reshape((self.Ny, self.Nx)), best_fitness
        return None, -1

    def run_post_analysis(self, initial_phi_path, theta_target_deg, phi_target_deg, vmin=None, vmax=None, phase_cmap='gray'):
        try:
            initial_phi = np.loadtxt(initial_phi_path, delimiter=',', dtype=int)
        except Exception as e:
            print(f"加载文件失败: {e}"); return
            
        print(f"正在为目标角度 (θ={theta_target_deg}, φ={phi_target_deg}) 计算最优列控密码 u...")
        u_vector = self.find_best_u(theta_target_deg, phi_target_deg, initial_phi)
        print(f"计算完成。最优 u = {u_vector}")
        
        final_phi = initial_phi.copy()
        for i, flip in enumerate(u_vector):
            if flip == 1: final_phi[:, i] = 1 - final_phi[:, i]
            
        # 更新实例的相位并计算完整方向图
        self.comp_phase_rad = self.xp.asarray(final_phi * np.pi)
        self.calculate_pattern(theta_points=181, phi_points=361)
        
        # MODIFIED: 调用父类中的公有可视化方法，并传递新参数
        self.plot_pattern_performance(theta_target_deg, phi_target_deg, vmin=vmin, vmax=vmax)
        self.visualize_phase_comparison(initial_phi, final_phi, cmap=phase_cmap)

if __name__ == '__main__':
    plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False
    M, N, P_MM, FREQ_HZ, R_FEED_M, HPBW_E_DEG, HPBW_H_DEG = 14, 14, 48, 2.95e9, 1.0, 20, 20
    GA_PARAMS = {'n_generations': 50, 'population_size': 80, 'mutation_rate': 0.05, 'crossover_rate': 0.8, 'elite_size': 4, 'n_jobs': -1}
    SCAN_ANGLES, PHI_PLANE, W_DEV, W_SLL = np.arange(-35.0, 35.1, 5.0), 0.0, 1.0, 0.15
    OPTIMIZED_PHASE_FILENAME = "optimized_phase_via_class.csv"
    
    my_array = ColControlledArray(n_elements_xy=(M, N), p=P_MM, freq=FREQ_HZ, r_feed=R_FEED_M, hpbw_e=HPBW_E_DEG, hpbw_h=HPBW_H_DEG, isdebug=1)
    
    print("\n" + "="*50 + "\n步骤 1: 开始全局优化初始相位...\n" + "="*50)
    optimized_phi, fitness = my_array.optimize_initial_phase(ga_params=GA_PARAMS, scan_angles_deg=SCAN_ANGLES, phi_obj_deg=PHI_PLANE, w_deviation=W_DEV, w_sll=W_SLL)

    if optimized_phi is not None:
        print(f"\n全局优化完成！最优适应度: {fitness:.6f}")
        # 尝试在脚本所在目录创建文件
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: # 在交互式环境中可能没有 __file__
            script_dir = os.getcwd()
        optimized_phase_path = os.path.join(script_dir, OPTIMIZED_PHASE_FILENAME)
        
        np.savetxt(optimized_phase_path, optimized_phi, delimiter=',', fmt='%d')
        print(f"最优初始相位已保存到: {optimized_phase_path}")
        
        print("\n" + "="*50 + "\n步骤 2: 使用优化后的相位进行后处理分析...\n" + "="*50)
        try:
            theta_in = float(input("请输入目标俯仰角 (例如, 25.0): "))
            phi_in = float(input("请输入目标方位角 (例如, 0.0): "))
        except ValueError:
            print("无效输入，使用默认值。"); theta_in, phi_in = 25.0, 0.0
        my_array.run_post_analysis(initial_phi_path=optimized_phase_path, theta_target_deg=theta_in, phi_target_deg=phi_in)
    else:
        print("\n全局优化失败。")