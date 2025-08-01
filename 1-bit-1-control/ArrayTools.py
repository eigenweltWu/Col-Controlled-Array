# 首先在文件开头添加GPU检测和相关库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import mplcursors
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='CUDA path could not be detected.*')

# 全局导入cupy，以便在类型提示和异常处理中使用
try:
    import cupy as cp
except ImportError:
    cp = None # 如果未安装cupy，则将其设置为None


class ArrayTools:
    def __init__(self, n_elements_xy, p, freq, r_feed, hpbw_e, hpbw_h, comp_phase_matrix, isdebug=0):
        """
        Args:
            n_elements_xy (tuple): 阵列在X和Y方向上的单元数量, e.g., (12, 16)。
            p (float): 单元间距 (mm)。
            freq (float): 工作频率 (Hz)。
            r_feed (float): 馈源喇叭与阵列中心的距离 (m)。
            hpbw_e (float): 馈源的 E 面半功率波束宽度 (度)。
            hpbw_h (float): 馈源的 H 面半功率波束宽度 (度)。
            comp_phase_matrix (np.array): 描述相位状态的0/1矩阵。
        """
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False

        self.isdebug = isdebug
        self.use_gpu = False
        self.xp = np

        # --- GPU/CPU 检测逻辑 (保持不变) ---
        try:
            import cupy as cp
            if cp and cp.cuda.is_available():
                self.use_gpu = True
                self.xp = cp
                if self.isdebug:
                    print("检测到GPU，将使用GPU加速")
            # ... (其他检测逻辑保持不变)
        except Exception as e:
            if self.isdebug:
                print(f"GPU检测失败: {str(e)}，将使用CPU计算")
            self.use_gpu = False
            self.xp = np
        
        if self.isdebug:
            print("ArrayTools基类已初始化。")

        # --- 初始化模拟器参数 (保持不变) ---
        self.Nx, self.Ny = n_elements_xy
        self.p_mm = p
        self.freq = freq
        self.r_feed_mm = r_feed * 1000
        self.hpbw_e_deg = hpbw_e
        self.hpbw_h_deg = hpbw_h
        
        if comp_phase_matrix.shape != (self.Ny, self.Nx):
            raise ValueError(f"提供的相位矩阵维度 {comp_phase_matrix.shape} 与阵列维度 {(self.Ny, self.Nx)} 不匹配。")
        
        self.comp_phase_rad = np.array(comp_phase_matrix) * np.pi

        self.c_mm_per_s = 3e11
        self.lambda_mm = self.c_mm_per_s / self.freq
        self.k = 2 * np.pi / self.lambda_mm

        # --- 修改：初始化结果存储变量 ---
        self.Gain = None
        self.AbsoluteGain_dBi = None  # 新增：用于存储绝对增益
        self.theta_scan_rad = None
        self.phi_scan_rad = None
        if self.isdebug:
            print(f"阵列模拟器参数初始化完成。阵列规模: {self.Nx}x{self.Ny}")

    def save_phase_array(self, filepath):
        """
        将当前的相位排布（0/pi 弧度）保存到CSV文件。

        Args:
            filepath (str): 保存文件的路径, e.g., 'phases.csv'。
        """
        # 确保数据在CPU上
        if self.use_gpu:
            phase_matrix_cpu = self.xp.asnumpy(self.comp_phase_rad)
        else:
            phase_matrix_cpu = self.comp_phase_rad
        
        # 将弧度转换为0/1格式以便存储
        phase_matrix_01 = (phase_matrix_cpu / np.pi).astype(int)
        
        try:
            np.savetxt(filepath, phase_matrix_01, delimiter=',', fmt='%d')
            if self.isdebug:
                print(f"相位排布已成功保存到: {os.path.abspath(filepath)}")
        except Exception as e:
            print(f"保存相位文件时出错: {e}")

    def load_phase_array(self, filepath):
        """
        从CSV文件加载相位排布。

        Args:
            filepath (str): 读取文件的路径, e.g., 'phases.csv'。
        """
        try:
            phase_matrix_01 = np.loadtxt(filepath, delimiter=',', dtype=int)
            
            if phase_matrix_01.shape != (self.Ny, self.Nx):
                raise ValueError(f"从文件加载的相位矩阵维度 {phase_matrix_01.shape} 与阵列维度 {(self.Ny, self.Nx)} 不匹配。")
            
            # 将0/1矩阵转换为0/pi弧度相位并更新
            self.comp_phase_rad = phase_matrix_01 * np.pi
            if self.isdebug:
                print(f"相位排布已成功从 {os.path.abspath(filepath)} 加载。")

        except FileNotFoundError:
            print(f"错误: 未找到文件 {filepath}")
        except Exception as e:
            print(f"加载相位文件时出错: {e}")

    def calculate_base_term(self, theta_obj_deg, phi_obj_deg=0):
        """
        预计算每个阵元在目标方向上的基础项（幅度和固定相位部分）
        Args:
            theta_obj_deg: 目标俯仰角 (度)
            phi_obj_deg: 目标方位角 (度), 默认为0
        Returns:
            base_term: 复数数组 (Ny, Nx)，返回CPU上的numpy数组
        """
        theta_obj = np.radians(theta_obj_deg)
        phi_obj = np.radians(phi_obj_deg)
        
        xp = self.xp # 使用基类中确定的计算后端

        # 1. 生成阵列网格
        x_1d = xp.arange(self.Nx) * self.p_mm - (self.Nx - 1) * self.p_mm / 2
        y_1d = xp.arange(self.Ny) * self.p_mm - (self.Ny - 1) * self.p_mm / 2
        x_grid, y_grid = xp.meshgrid(x_1d, y_1d)

        # 2. 馈源建模
        dist_to_feed = xp.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        hpbw_avg_rad = xp.radians((self.hpbw_e_deg + self.hpbw_h_deg) / 2)
        q = -3 / (20 * xp.log10(xp.cos(hpbw_avg_rad / 2)))
        theta_feed = xp.arccos(self.r_feed_mm / dist_to_feed)
        feed_pattern_amp = xp.cos(theta_feed)**q
        illumination_amp = feed_pattern_amp / dist_to_feed
        illumination_phase = -1 * self.k * dist_to_feed

        # 3. 空间相位延迟（目标方向）
        space_phase_delay = self.k * (x_grid * xp.sin(theta_obj) * xp.cos(phi_obj) + 
                                      y_grid * xp.sin(theta_obj) * xp.sin(phi_obj))

        # 4. 基础项 = 幅度 * exp(j*(馈源相位+初始补偿相位+空间相位延迟))
        base_term = illumination_amp * xp.exp(1j * (illumination_phase + xp.asarray(self.comp_phase_rad) + space_phase_delay))

        # 如果使用GPU，将结果转回CPU
        if self.use_gpu:
            return base_term.get()
        return base_term

    def calculate_pattern(self, theta_points=181, phi_points=361):
        """
        计算反射阵列的远场辐射方向图，并计算归一化增益和绝对增益。
        """
        if self.isdebug:
            print("开始计算远场方向图...")
        
        xp = self.xp
        
        # 1. 生成阵列网格 (保持不变)
        x_1d = xp.arange(self.Nx) * self.p_mm - (self.Nx - 1) * self.p_mm / 2
        y_1d = xp.arange(self.Ny) * self.p_mm - (self.Ny - 1) * self.p_mm / 2
        x_grid, y_grid = xp.meshgrid(x_1d, y_1d)

        # 2. 馈源建模 (保持不变)
        dist_to_feed = xp.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        hpbw_avg_rad = np.radians((self.hpbw_e_deg + self.hpbw_h_deg) / 2)
        q = -3 / (20 * np.log10(np.cos(hpbw_avg_rad / 2)))
        theta_feed = xp.arccos(self.r_feed_mm / dist_to_feed)
        feed_pattern_amp = xp.cos(theta_feed)**q
        illumination_amp = feed_pattern_amp / dist_to_feed
        illumination_phase = -1 * self.k * dist_to_feed
        comp_phase_rad_backend = xp.array(self.comp_phase_rad)

        # 3. 远场计算 (保持不变)
        self.theta_scan_rad = xp.linspace(-np.pi / 2, np.pi / 2, theta_points)
        self.phi_scan_rad = xp.linspace(-np.pi, np.pi, phi_points)
        
        if self.use_gpu:
            theta_grid_gpu, phi_grid_gpu = xp.meshgrid(self.theta_scan_rad, self.phi_scan_rad, indexing='ij')
            sin_theta = xp.sin(theta_grid_gpu)
            cos_phi = xp.cos(phi_grid_gpu)
            sin_phi = xp.sin(phi_grid_gpu)
            phase_shift = self.k * (x_grid[:, :, None, None] * sin_theta * cos_phi + 
                                    y_grid[:, :, None, None] * sin_theta * sin_phi)
            total_phase = illumination_phase[:, :, None, None] + comp_phase_rad_backend[:, :, None, None] + phase_shift
            E_field = xp.sum(illumination_amp[:, :, None, None] * xp.exp(1j * total_phase), axis=(0, 1))
        else:
            E_field = xp.zeros((theta_points, phi_points), dtype=complex)
            for i, theta_s in enumerate(self.theta_scan_rad):
                for j, phi_s in enumerate(self.phi_scan_rad):
                    far_field_phase_shift = self.k * (x_grid * np.sin(theta_s) * np.cos(phi_s) + 
                                                      y_grid * np.sin(theta_s) * np.sin(phi_s))
                    total_phase = illumination_phase + comp_phase_rad_backend + far_field_phase_shift
                    E_field[i, j] = xp.sum(illumination_amp * xp.exp(1j * total_phase))

        E_abs = xp.abs(E_field)
        E_max = xp.max(E_abs)

        # 4. 计算归一化增益 (dB)
        Gain_normalized = 20 * xp.log10(E_abs / E_max)
        Gain_normalized[Gain_normalized < -40] = -40
        self.Gain = Gain_normalized # 存储归一化增益

        # --- 修正：计算绝对增益/方向性 (dBi) ---
        # 在仰角(theta)从-pi/2到+pi/2的坐标系中，球面积分的面积元包含cos(theta)项
        theta_grid, _ = xp.meshgrid(self.theta_scan_rad, self.phi_scan_rad, indexing='ij')
        cos_theta_grid = xp.cos(theta_grid) # 修正：使用cos(theta)进行积分
        d_theta = self.theta_scan_rad[1] - self.theta_scan_rad[0]
        d_phi = self.phi_scan_rad[1] - self.phi_scan_rad[0]

        # 数值积分计算总辐射功率
        P_rad_integral = xp.sum(E_abs**2 * cos_theta_grid) * d_theta * d_phi

        # 计算方向性 (假设效率为100%，增益=方向性)
        if P_rad_integral > 1e-9: # 增加一个小的阈值以避免除以零
            Directivity = (4 * np.pi * E_max**2) / P_rad_integral
            self.AbsoluteGain_dBi = 10 * xp.log10(Directivity)
        else:
            self.AbsoluteGain_dBi = -xp.inf # 发生错误时的标记

        # 将结果转回CPU内存(numpy数组)供matplotlib使用
        if self.use_gpu:
            self.theta_scan_rad = self.xp.asnumpy(self.theta_scan_rad)
            self.phi_scan_rad = self.xp.asnumpy(self.phi_scan_rad)
            self.Gain = self.xp.asnumpy(self.Gain)
            self.AbsoluteGain_dBi = float(self.xp.asnumpy(self.AbsoluteGain_dBi)) # 确保是float类型
        else:
            self.theta_scan_rad = self.theta_scan_rad
            self.phi_scan_rad = self.phi_scan_rad
            self.Gain = self.Gain
        
        if self.isdebug:
            print(f"方向图计算完成。峰值绝对增益: {self.AbsoluteGain_dBi:.2f} dBi")
    
    def calculate_pattern_slice(self, phase_matrix_01, phi_deg, theta_points=181):
        """
        计算给定phi平面上的方向图切片。
        这是一个独立的计算函数，不修改实例的self.Gain等属性。
        
        Args:
            phase_matrix_01 (np.array): 用于计算的0/1相位矩阵。
            phi_deg (float): 需要计算的phi平面角度 (度)。
            theta_points (int): theta扫描点数。

        Returns:
            tuple: (gain_db, theta_deg)
        """
        xp = self.xp
        phi_s = np.radians(phi_deg)
        
        # 使用传入的相位矩阵
        current_phase_rad = xp.asarray(phase_matrix_01 * np.pi)
        
        # 阵列和馈源参数
        x_1d = xp.arange(self.Nx) * self.p_mm - (self.Nx - 1) * self.p_mm / 2
        y_1d = xp.arange(self.Ny) * self.p_mm - (self.Ny - 1) * self.p_mm / 2
        x_grid, y_grid = xp.meshgrid(x_1d, y_1d)
        
        dist_to_feed = xp.sqrt(x_grid**2 + y_grid**2 + self.r_feed_mm**2)
        hpbw_avg_rad = np.radians((self.hpbw_e_deg + self.hpbw_h_deg) / 2)
        q = -3 / (20 * np.log10(np.cos(hpbw_avg_rad / 2)))
        theta_feed = xp.arccos(xp.clip(self.r_feed_mm / dist_to_feed, -1.0, 1.0))
        illumination_amp = xp.cos(theta_feed)**q / dist_to_feed
        illumination_phase = -1 * self.k * dist_to_feed
        
        # 扫描特定平面
        theta_scan_rad = xp.linspace(-np.pi / 2, np.pi / 2, theta_points)
        E_field_slice = xp.zeros(theta_points, dtype=complex)
        
        for i, theta_s in enumerate(theta_scan_rad):
            far_field_phase_shift = self.k * (x_grid * xp.sin(theta_s) * np.cos(phi_s) + y_grid * xp.sin(theta_s) * np.sin(phi_s))
            total_phase = illumination_phase + current_phase_rad + far_field_phase_shift
            E_field_slice[i] = xp.sum(illumination_amp * xp.exp(1j * total_phase))
            
        E_abs = xp.abs(E_field_slice)
        E_max = xp.max(E_abs)
        if E_max == 0:
            if self.use_gpu:
                return np.full(theta_points, -100), np.degrees(xp.asnumpy(theta_scan_rad))
            else:
                return np.full(theta_points, -100), np.degrees(theta_scan_rad)
            
        Gain = 20 * xp.log10(E_abs / E_max)
        
        if self.use_gpu:
            return self.xp.asnumpy(Gain), np.degrees(self.xp.asnumpy(theta_scan_rad))
        else:
            return Gain, np.degrees(theta_scan_rad)

    def get_sidelobe_level(self, gain_db):
        """
        从一个增益切片中计算旁瓣电平 (SLL)。
        
        Args:
            gain_db (np.array): 归一化增益数组 (dB)。
            
        Returns:
            float: 最大旁瓣电平 (dB)。
        """
        # 寻找主瓣区域 (例如，高于-10dB的区域)
        main_lobe_indices = np.where(gain_db > -10)[0]
        if len(main_lobe_indices) == 0:
            return 0 # 没有明显主瓣

        # 创建一个掩码，排除主瓣区域
        sidelobe_mask = np.ones_like(gain_db, dtype=bool)
        sidelobe_mask[main_lobe_indices] = False
        
        sidelobes = gain_db[sidelobe_mask]
        
        return np.max(sidelobes) if len(sidelobes) > 0 else -100 # 如果没有旁瓣，返回一个很小的值

    def plot_pattern_performance(self, theta_target_deg, phi_target_deg, vmin=None, vmax=None):
        """
        可视化最终的波束性能，并在标题中显示绝对增益。
        需要先运行 `calculate_pattern`。

        Args:
            theta_target_deg (float): 目标俯仰角 (度)
            phi_target_deg (float): 目标方位角 (度)
            vmin (float, optional): 颜色条的最小值
            vmax (float, optional): 颜色条的最大值
        """
        # --- 修改：增加对AbsoluteGain_dBi的检查 ---
        if self.Gain is None or self.AbsoluteGain_dBi is None:
            raise ValueError("必须先调用 'calculate_pattern' 方法计算方向图。")
            
        fig = plt.figure(figsize=(16, 10))
        # --- 修改：在标题中加入绝对增益信息 ---
        fig.suptitle(f'最终波束方向图 (目标: θ={theta_target_deg}°, φ={phi_target_deg}° | 峰值绝对增益: {self.AbsoluteGain_dBi:.2f} dBi)', fontsize=16)

        # 1. 三维方向图 (保持不变)
        ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')
        Phi, Theta = np.meshgrid(self.phi_scan_rad, self.theta_scan_rad)
        R_offset = self.Gain - np.min(self.Gain)
        X, Y, Z = R_offset * np.sin(Theta) * np.cos(Phi), R_offset * np.sin(Theta) * np.sin(Phi), R_offset * np.cos(Theta)
        
        norm = Normalize(vmin=vmin if vmin is not None else np.min(self.Gain), 
                         vmax=vmax if vmax is not None else np.max(self.Gain))
                          
        ax1.plot_surface(X, Y, Z, facecolors=plt.cm.jet(norm(self.Gain)), rstride=2, cstride=2, antialiased=True, shade=False)
        ax1.set_title('三维方向图 (归一化)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 2. 顶视图 (pcolormesh) (保持不变)
        ax2 = fig.add_subplot(2, 2, 3)
        phi_deg, theta_deg = np.degrees(self.phi_scan_rad), np.degrees(self.theta_scan_rad)
        im = ax2.pcolormesh(theta_deg, phi_deg, self.Gain.T, cmap='jet', norm=norm, shading='auto')
        ax2.set_title('顶视图')
        ax2.set_xlabel('俯仰角 Theta (°)')
        ax2.set_ylabel('方位角 Phi (°)')
        ax2.set_xlim(-90, 90)
        ax2.set_ylim(-180, 180)
        fig.colorbar(im, ax=ax2, label='归一化增益 (dB)', pad=0.1)
        
        # 3. 目标平面切片图 (保持不变)
        ax3 = fig.add_subplot(2, 2, 4)
        phi_idx = np.argmin(np.abs(phi_deg - phi_target_deg))
        gain_slice = self.Gain[:, phi_idx]
        peak_angle = theta_deg[np.argmax(gain_slice)]
        
        ax3.plot(theta_deg, gain_slice, label=f'实际峰值: {peak_angle:.2f}°')
        ax3.axvline(x=theta_target_deg, color='r', linestyle='--', label=f'目标角度: {theta_target_deg}°')
        ax3.set_title(f'目标平面切片 (Phi ≈ {phi_deg[phi_idx]:.1f}°)')
        ax3.set_xlabel('俯仰角 Theta (°)')
        ax3.set_ylabel('归一化增益 (dB)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_ylim(-40, 1)
        ax3.set_xlim(-90, 90)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_phase_comparison(self, initial_phi, final_phi, cmap='gray'):
        """
        并排可视化初始相位和最终相位矩阵。
        
        Args:
            initial_phi (np.array): 初始0/1相位矩阵。
            final_phi (np.array): 最终0/1相位矩阵。
            cmap (str, optional): 颜色映射名称，默认为'gray'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('相位分布对比', fontsize=16)
        
        ax1.imshow(initial_phi, cmap=cmap, interpolation='nearest')
        ax1.set_title('初始相位 Φ_mn')
        ax1.set_xlabel('列 (M)')
        ax1.set_ylabel('行 (N)')
        
        im = ax2.imshow(final_phi, cmap=cmap, interpolation='nearest')
        ax2.set_title('最终相位')
        ax2.set_xlabel('列 (M)')

        # 添加颜色条
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, ticks=[0, 1], label='相位状态 (0: 0°, 1: 180°)')
        
        plt.show()

    # =========================================================================
    
    def plot_3d_pattern(self):
        """
        绘制三维辐射方向图及附加视图(phi=0, phi=90和顶视图)，并启用交互式数据提示。
        """
        if self.Gain is None:
            raise ValueError("必须先调用 'calculate_pattern' 方法计算方向图。")
        
        if self.isdebug:
            print("开始绘制三维方向图及附加视图...")
        # 准备绘图数据
        Phi, Theta = np.meshgrid(self.phi_scan_rad, self.theta_scan_rad)
        
        # 对增益进行偏移以便于在三维中绘图
        R_offset = self.Gain - np.min(self.Gain)
    
        # 球坐标到笛卡尔坐标转换（用于底层绘图）
        X = R_offset * np.sin(Theta) * np.cos(Phi)
        Y = R_offset * np.sin(Theta) * np.sin(Phi)
        Z = R_offset * np.cos(Theta)
    
        # 优化：减少绘图点数以降低内存使用
        step = 1
        if X.size > 5000:
            step = max(1, int(np.sqrt(X.size / 5000)))
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]
        Z_sub = Z[::step, ::step]
        Gain_subsampled = self.Gain[::step, ::step]
        self.step = step
    
        # 调整图形大小
        fig = plt.figure(figsize=(12, 10))
    
        # 1. 三维方向图
        ax1 = fig.add_subplot(221, projection='3d')
        norm = Normalize(vmin=np.min(self.Gain), vmax=np.max(self.Gain))
        surface = ax1.plot_surface(X_sub, Y_sub, Z_sub, facecolors=plt.cm.jet(norm(Gain_subsampled)), 
                                  rstride=1, cstride=1, antialiased=True, shade=False)
        ax1.set_title('三维方向图')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_box_aspect((np.ptp(X_sub), np.ptp(Y_sub), np.ptp(Z_sub)))
    
        # 2. phi=0方向图
        ax2 = fig.add_subplot(222)
        phi0_idx = np.argmin(np.abs(self.phi_scan_rad))
        theta_deg = np.degrees(self.theta_scan_rad)
        ax2.plot(theta_deg, self.Gain[:, phi0_idx])
        ax2.set_title('phi=0° 方向图 (E面)')
        ax2.set_xlabel('θ角度 (°)')
        ax2.set_ylabel('增益 (dB)')
        ax2.grid(True)
    
        # 3. phi=90方向图
        ax3 = fig.add_subplot(223)
        phi90_idx = np.argmin(np.abs(self.phi_scan_rad - np.pi/2))
        ax3.plot(theta_deg, self.Gain[:, phi90_idx])
        ax3.set_title('phi=90° 方向图 (H面)')
        ax3.set_xlabel('θ角度 (°)')
        ax3.set_ylabel('增益 (dB)')
        ax3.grid(True)
    
        # 4. 顶视图
        ax4 = fig.add_subplot(224)
        phi_deg = np.degrees(self.phi_scan_rad)
        im = ax4.pcolormesh(theta_deg, phi_deg, self.Gain.T, cmap='jet', norm=norm)
        ax4.set_title('顶视图 (增益分布)')
        ax4.set_xlabel('θ角度 (°)')
        ax4.set_ylabel('φ角度 (°)')
        ax4.set_xlim(-90, 90)
        ax4.set_ylim(-180, 180)
        ax4.grid(True)
    
        # 颜色条
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        fig.colorbar(im, cax=cbar_ax, label='增益 (dB)')
    
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # 交互式数据提示
        cursor = mplcursors.cursor(surface, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            row, col = sel.index
            orig_row = row * self.step
            orig_col = col * self.step
            theta_val_deg = np.rad2deg(self.theta_scan_rad[orig_row])
            phi_val_deg = np.rad2deg(self.phi_scan_rad[orig_col])
            gain_val_db = self.Gain[orig_row, orig_col]
            sel.annotation.set_text(f'Theta: {theta_val_deg:.1f}°\nPhi: {phi_val_deg:.1f}°\nGain: {gain_val_db:.2f} dB')
        plt.show()

    def run_genetic_optimization(self, fitness_function, initial_population, ga_params):
        """
        一个通用的遗传算法模板，用于优化阵列参数。
        该算法利用joblib进行并行适应度评估，并允许适应度函数内部利用GPU加速。

        Args:
            fitness_function (callable): 适应度函数。
                它应接受两个参数: (chromosome, simulator_instance)
                并返回一个浮点数作为适应度得分。分数越高越好。
            initial_population (np.array): 初始种群，一个2D numpy数组，每行代表一个个体（染色体）。
            ga_params (dict): 包含遗传算法参数的字典。
                - 'n_generations' (int): 迭代总代数。
                - 'population_size' (int): 种群大小。
                - 'mutation_rate' (float): 变异率。
                - 'crossover_rate' (float): 交叉率。
                - 'elite_size' (int): 精英个体数量。
                - 'n_jobs' (int): joblib并行任务数, -1表示使用所有CPU核心。

        Returns:
            tuple: (best_chromosome, best_fitness)
                - best_chromosome (np.array): 找到的最优个体（染色体）。
                - best_fitness (float): 最优个体的适应度得分。
        """
        population = np.copy(initial_population)
        best_chromosome_overall = None
        best_fitness_overall = -float('inf')
        
        # 从ga_params字典解包参数
        n_generations = ga_params.get('n_generations', 100)
        population_size = ga_params.get('population_size', 50)
        mutation_rate = ga_params.get('mutation_rate', 0.05)
        crossover_rate = ga_params.get('crossover_rate', 0.8)
        elite_size = ga_params.get('elite_size', 5)
        n_jobs = ga_params.get('n_jobs', -1)

        print("\n--- 开始遗传算法优化 ---")
        start_time = os.times().user
        
        for generation in range(n_generations):
            # --- 1. 适应度评估 (并行) ---
            # delayed(fitness_function)创建了一个延迟执行的版本
            # Parallel(...)会并行地对种群中每个染色体调用这个函数
            with Parallel(n_jobs=n_jobs) as parallel:
                fitness_scores = parallel(
                    delayed(fitness_function)(chromo, self) for chromo in tqdm(
                        population, desc=f"第 {generation + 1}/{n_generations} 代评估", leave=False
                    )
                )
            fitness_scores = np.array(fitness_scores)

            # --- 2. 寻找最优解 ---
            best_idx_gen = np.argmax(fitness_scores)
            if fitness_scores[best_idx_gen] > best_fitness_overall:
                best_fitness_overall = fitness_scores[best_idx_gen]
                best_chromosome_overall = population[best_idx_gen].copy()
                print(f"\n第 {generation + 1} 代: 发现新最优解! 适应度: {best_fitness_overall:.6f}")

            # --- 3. 选择 (精英主义 + 锦标赛选择) ---
            sorted_indices = np.argsort(fitness_scores)[::-1]
            
            # 精英主义：直接保留最优的个体
            new_population = [population[i] for i in sorted_indices[:elite_size]]
            
            # 锦标赛选择：为剩余的个体选择父母
            num_parents_needed = population_size - elite_size
            tournament_size = 5
            parent_indices = []
            for _ in range(num_parents_needed):
                 # 随机选择k个个体进行锦标赛
                tournament_contenders_indices = np.random.choice(range(population_size), tournament_size, replace=False)
                # 在锦标赛中选择适应度最高的作为父代
                winner_index_in_tournament = np.argmax(fitness_scores[tournament_contenders_indices])
                parent_indices.append(tournament_contenders_indices[winner_index_in_tournament])

            parents = population[parent_indices]

            # --- 4. 交叉 ---
            offspring = []
            for i in range(0, num_parents_needed, 2):
                if i + 1 >= len(parents): # 如果父代数量为奇数
                    offspring.append(parents[i])
                    continue
                
                p1, p2 = parents[i], parents[i+1]
                if np.random.rand() < crossover_rate:
                    # 单点交叉
                    crossover_point = np.random.randint(1, len(p1) - 1)
                    c1 = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
                    c2 = np.concatenate([p2[:crossover_point], p1[crossover_point:]])
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])

            # --- 5. 变异 ---
            # 假设染色体是二进制的，进行位翻转变异
            for i in range(len(offspring)):
                mutation_mask = np.random.rand(len(offspring[i])) < mutation_rate
                offspring[i][mutation_mask] = 1 - offspring[i][mutation_mask]

            new_population.extend(offspring)
            population = np.array(new_population[:population_size])

        end_time = os.times().user
        print(f"\n优化完成！总耗时: {(end_time - start_time):.2f} 秒")
        
        return best_chromosome_overall, best_fitness_overall


