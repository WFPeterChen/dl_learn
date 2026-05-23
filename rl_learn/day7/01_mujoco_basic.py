"""day7 / 01: MuJoCo Python API 直接调用 (不通过 gym 包装).

目的:
  剥开 Gymnasium 这层包装, 直接用 mujoco library 控制 sim.
  这是 day10 C++ 部署的 Python 原型 — 一旦能用纯 mujoco API 跑控制 loop,
  迁移到 C++ MuJoCo API 就只是语法翻译.

对比:
  gym 包装版:   obs, r, term, trunc, _ = env.step(action)
                ↑                              ↑
              gym 自动算的                   gym 自动算的

  mujoco 直接:  data.ctrl[:] = action
                mujoco.mj_step(model, data)
                obs = concat(data.qpos, data.qvel)
                r   = ... (你自己定义)
                term = ... (你自己定义)

  → gym 帮你做了"reward / terminate / obs 提取"的封装.
    直接用 mujoco 需要自己定义这些.

本文件 demo:
  ① explore_model:    打印 model 的 bodies / joints / geoms / actuators
  ② basic_step_demo:  无控制, 杆在重力下自由倒下
  ③ pd_controller:    经典 PD 控制器稳定杆 (不用 NN, 纯解析控制)
"""
import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt

# MJCF 路径 — 复用 gymnasium 自带的 InvertedPendulum
import gymnasium.envs.mujoco
MJCF_PATH = os.path.join(
    os.path.dirname(gymnasium.envs.mujoco.__file__),
    "assets", "inverted_pendulum.xml",
)


# ====================================================================
# ① 探索 model
# ====================================================================
def explore_model(model):
    print(f"=== Model from {os.path.basename(MJCF_PATH)} ===")
    print(f"  nbody    = {model.nbody}        (worldbody + 各 named bodies)")
    print(f"  njnt     = {model.njnt}        (joints)")
    print(f"  ngeom    = {model.ngeom}        (geoms)")
    print(f"  nu       = {model.nu}        (actuators = action dim)")
    print(f"  nq       = {model.nq}        (qpos dim, 位置自由度)")
    print(f"  nv       = {model.nv}        (qvel dim, 速度自由度)")
    print(f"  timestep = {model.opt.timestep}s ({1/model.opt.timestep:.0f} Hz)")

    print(f"\n  Bodies:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(world)"
        print(f"    [{i}] {name}")

    print(f"\n  Joints:")
    JNT_TYPES = ["free", "ball", "slide", "hinge"]
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    [{i}] {name}  (type={JNT_TYPES[model.jnt_type[i]]}, range={model.jnt_range[i]})")

    print(f"\n  Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        cr = model.actuator_ctrlrange[i]
        gear = model.actuator_gear[i, 0]
        print(f"    [{i}] {name}  (ctrl range={cr}, gear={gear})")


# ====================================================================
# ② 无控制: 重力下杆倒下 (sanity check 物理引擎在动)
# ====================================================================
def free_fall_demo(model, data, n_steps=50):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.087    # ≈ 5 度初始小角度
    mujoco.mj_forward(model, data)   # 只重算依赖量 (不 step)

    history = {"t": [], "x": [], "theta": []}
    for _ in range(n_steps):
        # data.ctrl 默认 0, 无控制
        mujoco.mj_step(model, data)   # 推进一步物理
        history["t"].append(data.time)
        history["x"].append(data.qpos[0])
        history["theta"].append(data.qpos[1])

    print(f"\n=== Free fall demo (no ctrl, gravity only) ===")
    print(f"  跑了 {n_steps} 步 = {n_steps * model.opt.timestep:.2f}s")
    print(f"  theta: {history['theta'][0]:+.3f} → {history['theta'][-1]:+.3f} rad")
    print(f"  x:     {history['x'][0]:+.3f} → {history['x'][-1]:+.3f} m   (cart 反作用力推开)")
    return history


# ====================================================================
# ③ PD controller demo — 经典控制论, 不用 NN
# ====================================================================
def pd_controller_demo(model, data, n_steps=500, K_p=30.0, K_d=5.0):
    """简单 PD: ctrl = K_p · θ + K_d · θ_dot.

    直觉: 杆向右倒 (θ>0) → cart 也往右推 (ctrl>0) → cart 加速 → 反作用力扶正杆
          θ_dot 项 damping (阻尼), 防止震荡.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.05    # 小初始扰动
    mujoco.mj_forward(model, data)

    history = {"t": [], "x": [], "theta": [], "u": []}
    for _ in range(n_steps):
        theta     = data.qpos[1]
        theta_dot = data.qvel[1]
        u = K_p * theta + K_d * theta_dot
        u = float(np.clip(u, -3.0, 3.0))    # ctrl range 是 [-3, 3]
        data.ctrl[0] = u

        mujoco.mj_step(model, data)

        history["t"].append(data.time)
        history["x"].append(data.qpos[0])
        history["theta"].append(data.qpos[1])
        history["u"].append(u)

        if abs(data.qpos[1]) > 0.2:   # gym 的 terminate 条件
            print(f"  ⚠ terminated at t={data.time:.2f}s (|θ| > 0.2)")
            break

    print(f"\n=== PD controller (K_p={K_p}, K_d={K_d}) ===")
    print(f"  survived: {len(history['t'])} steps ({len(history['t']) * model.opt.timestep:.2f}s)")
    print(f"  final theta: {history['theta'][-1]:+.4f} rad")
    print(f"  final x:     {history['x'][-1]:+.3f} m")
    return history


# ====================================================================
# 可视化
# ====================================================================
def plot(history_free, history_pd, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    ax = axes[0, 0]
    ax.plot(history_free["t"], history_free["theta"], label="no ctrl (free fall)", color='red')
    ax.plot(history_pd["t"],   history_pd["theta"],   label="PD control", color='blue')
    ax.axhline(0.2, ls='--', c='gray', alpha=0.5); ax.axhline(-0.2, ls='--', c='gray', alpha=0.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("θ (rad)")
    ax.set_title("Pole angle θ"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history_free["t"], history_free["x"], label="no ctrl", color='red')
    ax.plot(history_pd["t"],   history_pd["x"],   label="PD control", color='blue')
    ax.set_xlabel("t (s)"); ax.set_ylabel("x (m)")
    ax.set_title("Cart position x"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history_pd["t"], history_pd["u"], color='purple')
    ax.set_xlabel("t (s)"); ax.set_ylabel("ctrl")
    ax.set_title("PD control output"); ax.grid(alpha=0.3)
    ax.axhline(3,  ls='--', c='r', alpha=0.4, label='ctrl saturation')
    ax.axhline(-3, ls='--', c='r', alpha=0.4)
    ax.legend()

    axes[1, 1].axis('off')
    axes[1, 1].text(0.05, 0.7,
        "Demo 1 (red):  无 ctrl, 重力下杆倒\n"
        "Demo 2 (blue): PD ctrl = K_p·θ + K_d·θ_dot\n"
        "                能稳定 ~10s 直到 cart 撞墙\n\n"
        "下一步: 用 day6 训好的 DQN policy\n"
        "        替换 PD 公式 (02_mujoco_dqn.py)",
        fontsize=10, family='monospace')

    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ====================================================================
# Main
# ====================================================================
def main():
    print(f"Loading MJCF: {MJCF_PATH}\n")

    # 加载模型 + 创建 data
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)

    # ① 探索
    explore_model(model)

    # ② 无控制
    history_free = free_fall_demo(model, data, n_steps=50)

    # ③ PD 控制
    history_pd = pd_controller_demo(model, data, n_steps=500, K_p=30.0, K_d=5.0)

    # 出图
    figs_dir = os.path.join(os.path.dirname(__file__) or ".", "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plot(history_free, history_pd, os.path.join(figs_dir, "01_basic_demo.png"))
    print(f"\n✓ figs → figs/01_basic_demo.png")


if __name__ == "__main__":
    main()
