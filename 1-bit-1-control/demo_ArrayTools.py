import numpy as np
import os
from ArrayTools import ArrayTools

# =========================================================================
#  遗传算法适应度函数 (特定于此演示)
# =========================================================================
def maximize_gain_at_angle(chromosome, simulator):
    """
    一个适应度函数示例：优化相位矩阵以最大化指定方向的增益。
    
    Args:
        chromosome (np.array): 一个一维的0/1数组，代表展平的相位矩阵。
        simulator (ArrayTools): ArrayTools类的实例。
        
    Returns:
        float: 在目标方向上的绝对场强（作为适应度）。
    """
    # 1. 解码染色体
    phase_matrix_01 = chromosome.reshape((simulator.Ny, simulator.Nx))
    
    xp = simulator.xp
    temp_comp_phase_rad = xp.asarray(phase_matrix_01 * np.pi)
    
    # --- 复用核心逻辑进行快速评估 ---
    target_theta_rad = xp.radians(20.0) # 优化目标角度
    target_phi_rad = xp.radians(0.0)

    x_1d = xp.arange(simulator.Nx) * simulator.p_mm - (simulator.Nx - 1) * simulator.p_mm / 2
    y_1d = xp.arange(simulator.Ny) * simulator.p_mm - (simulator.Ny - 1) * simulator.p_mm / 2
    x_grid, y_grid = xp.meshgrid(x_1d, y_1d)

    dist_to_feed = xp.sqrt(x_grid**2 + y_grid**2 + simulator.r_feed_mm**2)
    hpbw_avg_rad = np.radians((simulator.hpbw_e_deg + simulator.hpbw_h_deg) / 2)
    q = -3 / (20 * np.log10(np.cos(hpbw_avg_rad / 2)))
    theta_feed = xp.arccos(simulator.r_feed_mm / dist_to_feed)
    feed_pattern_amp = xp.cos(theta_feed)**q
    illumination_amp = feed_pattern_amp / dist_to_feed
    illumination_phase = -1 * simulator.k * dist_to_feed
    
    far_field_phase_shift = simulator.k * (x_grid * xp.sin(target_theta_rad) * xp.cos(target_phi_rad) + 
                                           y_grid * xp.sin(target_theta_rad) * xp.sin(target_phi_rad))
    
    total_phase = illumination_phase + temp_comp_phase_rad + far_field_phase_shift
    E_field_at_target = xp.sum(illumination_amp * xp.exp(1j * total_phase))
    
    fitness = xp.abs(E_field_at_target)
    
    return float(fitness.get()) if simulator.use_gpu else float(fitness)

# =========================================================================
#  主执行脚本
# =========================================================================
if __name__ == '__main__':
    # --- 1. 定义阵列参数 ---
    NX, NY = 12, 10
    P_MM = 45
    FREQ_HZ = 2.97e9
    R_FEED_M = 1
    HPBW_E_DEG = 10
    HPBW_H_DEG = 10
    
    # 创建一个随机的初始相位矩阵
    INITIAL_PHASE_MATRIX_01 = np.random.randint(0, 2, size=(NY, NX))

    # --- 2. 初始化阵列模拟器 ---
    print("--- 初始化 ArrayTools 演示 ---")
    array_instance = ArrayTools(
        n_elements_xy=(NX, NY), 
        p=P_MM, 
        freq=FREQ_HZ, 
        r_feed=R_FEED_M, 
        hpbw_e=HPBW_E_DEG, 
        hpbw_h=HPBW_H_DEG, 
        comp_phase_matrix=INITIAL_PHASE_MATRIX_01,
        isdebug=1
    )

    # --- 3. 演示遗传算法优化 ---
    print("\n--- 演示遗传算法 ---")
    ga_parameters = {
        'n_generations': 20,
        'population_size': 40,
        'mutation_rate': 0.02,
        'crossover_rate': 0.8,
        'elite_size': 4,
        'n_jobs': -1, # 使用所有CPU核心
    }

    initial_pop = np.random.randint(0, 2, size=(ga_parameters['population_size'], NY * NX))

    best_phase_flat, best_fitness = array_instance.run_genetic_optimization(
        fitness_function=maximize_gain_at_angle,
        initial_population=initial_pop,
        ga_params=ga_parameters
    )

    # --- 4. 显示优化结果 ---
    print("\n--- 优化结果 ---")
    if best_phase_flat is not None:
        print(f"找到的最优适应度: {best_fitness}")
        optimized_phase_matrix_01 = best_phase_flat.reshape((NY, NX))
        array_instance.comp_phase_rad = optimized_phase_matrix_01 * np.pi
        
        print("最优相位排布已应用，正在计算并显示最终方向图...")
        # 添加 vmin 和 vmax 参数控制 color bar 范围
        array_instance.calculate_pattern()
        array_instance.plot_pattern_performance(theta_target_deg=20.0, phi_target_deg=0.0, vmin=-30, vmax=0)
        
        print("显示初始相位与优化后相位的对比...")
        # 添加 cmap 参数自定义颜色映射
        array_instance.visualize_phase_comparison(INITIAL_PHASE_MATRIX_01, optimized_phase_matrix_01, cmap='viridis')

        # 保存最优结果
        output_filename = 'demo_optimized_pattern.csv'
        array_instance.save_phase_array(output_filename)
        print(f"演示的最优相位图已保存至: {output_filename}")
    else:
        print("未能找到最优解。")