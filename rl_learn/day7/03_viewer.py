"""day7 / 03: MuJoCo 交互式 viewer.

直接在 dl-box 本地终端跑 (不要从主开发机 ssh!), 会弹出 GUI 窗口.

用法:
    python 03_viewer.py --mode dqn      # 用 day6 训好的 DQN policy (默认)
    python 03_viewer.py --mode pd       # PD controller (会撞墙 terminate, 自动 reset)
    python 03_viewer.py --mode free     # 不控制, 杆自由倒下

窗口操作:
    左键拖动     旋转视角
    右键拖动     平移视角
    滚轮         缩放
    Esc / 关窗   退出
    Space        暂停 / 继续 (viewer 自带)
    F1           帮助
"""
import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# 让 import day6 的 train_dqn 模块可用
sys.path.insert(0, os.path.join(os.path.dirname(__file__) or ".", "..", "day6"))

# MJCF 路径
import gymnasium.envs.mujoco
MJCF_PATH = os.path.join(
    os.path.dirname(gymnasium.envs.mujoco.__file__),
    "assets", "inverted_pendulum.xml",
)


# ====================================================================
# 三种控制策略, 统一接口: fn(data) -> ctrl (float)
# ====================================================================
def make_free_ctrl():
    return lambda data: 0.0


def make_pd_ctrl(K_p=30.0, K_d=5.0):
    def pd(data):
        theta, theta_dot = data.qpos[1], data.qvel[1]
        return float(np.clip(K_p * theta + K_d * theta_dot, -3.0, 3.0))
    return pd


def make_dqn_ctrl():
    import torch
    from train_dqn import QNet, ACTION_VALUES, DEVICE

    net = QNet().to(DEVICE)
    pt_path = os.path.join(os.path.dirname(__file__) or ".", "..", "day6", "trained_dqn.pt")
    net.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    net.eval()
    print(f"  loaded DQN from {pt_path}  (device={DEVICE}, {sum(p.numel() for p in net.parameters())} params)")

    def dqn(data):
        obs = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a = int(net(s).argmax(dim=1).item())
        return ACTION_VALUES[a]
    return dqn


# ====================================================================
# 主循环 — passive viewer + 自己 step 物理
# ====================================================================
def run_viewer(ctrl_fn, init_theta=0.05, real_time=True, auto_reset=True):
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)
    data.qpos[1] = init_theta
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep   # 0.02s = 50 Hz

    # passive viewer: 不阻塞 — 我们自己跑 sim loop, viewer 只是显示
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n窗口已打开. 关窗 / Esc 退出.\n")
        while viewer.is_running():
            tic = time.time()

            # 控制 + step
            data.ctrl[0] = ctrl_fn(data)
            mujoco.mj_step(model, data)

            # 让 viewer 拿到最新 data
            viewer.sync()

            # auto reset on terminate (方便观察反复试)
            if auto_reset and abs(data.qpos[1]) > 0.2:
                print(f"  terminated at t={data.time:.2f}s, resetting...")
                mujoco.mj_resetData(model, data)
                data.qpos[1] = init_theta
                mujoco.mj_forward(model, data)

            # real-time 速度 (不然 50 Hz 物理一秒跑完会 instant skip)
            if real_time:
                elapsed = time.time() - tic
                if elapsed < dt:
                    time.sleep(dt - elapsed)


# ====================================================================
# CLI
# ====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["free", "pd", "dqn"], default="dqn",
                   help="控制策略: free=无控制 / pd=经典 PD / dqn=day6 训好的 NN policy")
    p.add_argument("--init_theta", type=float, default=0.05,
                   help="初始杆角度 (rad)")
    p.add_argument("--no_realtime", action="store_true",
                   help="跑得越快越好 (调试用; 默认 real-time)")
    p.add_argument("--no_reset", action="store_true",
                   help="terminate 后不 reset")
    args = p.parse_args()

    print(f"=== MuJoCo viewer: mode={args.mode} ===")
    if args.mode == "free":
        ctrl_fn = make_free_ctrl()
    elif args.mode == "pd":
        ctrl_fn = make_pd_ctrl()
    elif args.mode == "dqn":
        ctrl_fn = make_dqn_ctrl()

    run_viewer(
        ctrl_fn,
        init_theta=args.init_theta,
        real_time=not args.no_realtime,
        auto_reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
