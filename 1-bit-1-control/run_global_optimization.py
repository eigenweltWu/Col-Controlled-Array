import numpy as np
import os
from ColControlledArray import ColControlledArray

if __name__ == '__main__':
    # =========================================================================
    #  1. 定义仿真和优化参数
    # =========================================================================
    
    # --- 阵列物理参数 ---
    M, N = 14, 14
    P_MM = 48
    FREQ_HZ = 2.95e9
    R_FEED_M = 1.0
    HPBW_E_DEG = 20
    HPBW_H_DEG = 20
    
    # --- 遗传算法参数 ---
    GA_PARAMS = {
        'n_generations': 50,    # 迭代代数
        'population_size': 80,  # 种群大小
        'mutation_rate': 0.05,  # 变异率
        'crossover_rate': 0.8,  # 交叉率
        'elite_size': 4,        # 精英个体数
        'n_jobs': -1,           # 使用所有CPU核心并行计算
    }
    
    # --- 优化目标参数 ---
    # 定义扫描角度范围、评估的phi平面以及各项的权重
    SCAN_ANGLES = np.arange(-35.0, 35.1, 5.0)  # 扫描角度范围从-35到+35度，步进5度
    PHI_PLANE = 0.0                             # 在phi=0平面上评估性能
    W_DEV = 1.0                                 # 波束指向偏差的惩罚权重
    W_SLL = 0.15                                # 旁瓣电平的惩罚权重
    
    # --- 输出文件名 ---
    OPTIMIZED_PHASE_FILENAME = "optimized_initial_phase.csv"

    # =========================================================================
    #  2. 执行全局优化
    # =========================================================================
    
    print("--- 初始化列控阵列 (ColControlledArray) ---")
    my_array = ColControlledArray(
        n_elements_xy=(M, N), 
        p=P_MM, 
        freq=FREQ_HZ, 
        r_feed=R_FEED_M, 
        hpbw_e=HPBW_E_DEG, 
        hpbw_h=HPBW_H_DEG, 
        isdebug=0
    )
    
    print("\n" + "="*60)
    print("步骤 1: 开始全局优化初始相位 Φ_mn...")
    print(f"扫描角度范围: {SCAN_ANGLES}")
    print(f"GA 参数: {GA_PARAMS}")
    print("这个过程可能需要较长时间，请耐心等待...")
    print("="*60)
    start_time = time.time()
    # 运行优化过程
    optimized_phi, fitness = my_array.optimize_initial_phase(
        ga_params=GA_PARAMS, 
        scan_angles_deg=SCAN_ANGLES, 
        phi_obj_deg=PHI_PLANE, 
        w_deviation=W_DEV, 
        w_sll=W_SLL
    )

    # =========================================================================
    #  3. 保存结果
    # =========================================================================
    end_time = time.time()
    print(f"全局优化耗时: {end_time - start_time:.2f} 秒")
    
    if optimized_phi is not None:
        print(f"\n全局优化完成！最优适应度: {fitness:.6f}")
        
        # 确定文件保存路径
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: 
            script_dir = os.getcwd()
        optimized_phase_path = os.path.join(script_dir, OPTIMIZED_PHASE_FILENAME)
        
        # 保存到CSV文件
        np.savetxt(optimized_phase_path, optimized_phi, delimiter=',', fmt='%d')
        print(f"\n最优初始相位已成功保存到: {optimized_phase_path}")
        print("现在可以运行 'run_post_processing.py' 脚本来进行后处理分析。")
    else:
        print("\n全局优化失败，未能找到有效解。")