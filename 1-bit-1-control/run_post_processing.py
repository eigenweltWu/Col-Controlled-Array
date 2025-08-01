import numpy as np
import os
from ColControlledArray import ColControlledArray

if __name__ == '__main__':
    # =========================================================================
    #  1. 定义与优化时一致的仿真参数
    # =========================================================================
    
    # --- 阵列物理参数 (必须与优化时使用的参数完全相同) ---
    M, N = 14, 14
    P_MM = 48
    FREQ_HZ = 2.95e9
    R_FEED_M = 1.0
    HPBW_E_DEG = 20
    HPBW_H_DEG = 20
    
    # --- 输入文件名 (必须与优化脚本的输出文件名相同) ---
    OPTIMIZED_PHASE_FILENAME = "optimized_initial_phase.csv"

    # =========================================================================
    #  2. 执行后处理分析
    # =========================================================================

    print("--- 初始化列控阵列 (ColControlledArray) 以进行后处理 ---")
    my_array = ColControlledArray(
        n_elements_xy=(M, N), 
        p=P_MM, 
        freq=FREQ_HZ, 
        r_feed=R_FEED_M, 
        hpbw_e=HPBW_E_DEG, 
        hpbw_h=HPBW_H_DEG, 
        isdebug=0
    )

    # 检查优化后的相位文件是否存在
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    optimized_phase_path = os.path.join(script_dir, OPTIMIZED_PHASE_FILENAME)

    if not os.path.exists(optimized_phase_path):
        print(f"\n错误: 未找到优化后的相位文件 '{OPTIMIZED_PHASE_FILENAME}'。")
        print("请先运行 'run_global_optimization.py' 脚本来生成此文件。")
    else:
        print(f"\n成功找到相位文件: {optimized_phase_path}")
        print("\n" + "="*60)
        print("步骤 2: 开始后处理分析...")
        print("="*60)

        # 获取用户输入的目标角度
        try:
            theta_in = float(input("请输入目标俯仰角 (θ, e.g., 25.0): "))
            phi_in = float(input("请输入目标方位角 (φ, e.g., 0.0): "))
        except ValueError:
            print("无效输入，将使用默认值 (θ=25.0°, φ=0.0°)。")
            theta_in, phi_in = 25.0, 0.0

        # 运行后处理分析
        my_array.run_post_analysis(
            initial_phi_path=optimized_phase_path, 
            theta_target_deg=theta_in, 
            phi_target_deg=phi_in
        )