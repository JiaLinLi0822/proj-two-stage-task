import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Circle, FancyArrowPatch

# -------------------- 工具函数：将行为 JSON 规范为 {trial_index: trial_dict} --------------------
def build_trial_map(behavior_obj):
    """
    规范化为 {trial_index: trial_dict}。
    支持：
      - list[trial_dict]
      - 单个 trial_dict（含 "trial_index"）
      - dict[str/int -> trial_dict]（键为 trial id）
      - dict 含 "trials": list[trial_dict]
    """
    if isinstance(behavior_obj, list):
        m = {int(it["trial_index"]): it for it in behavior_obj if isinstance(it, dict) and "trial_index" in it}
        if m: return m

    if isinstance(behavior_obj, dict):
        if "trial_index" in behavior_obj and "graph" in behavior_obj:
            return {int(behavior_obj["trial_index"]): behavior_obj}
        if "trials" in behavior_obj and isinstance(behavior_obj["trials"], list):
            m = {int(it["trial_index"]): it for it in behavior_obj["trials"] if isinstance(it, dict) and "trial_index" in it}
            if m: return m
        # 尝试把 key 当作 trial id
        m = {}
        convertible = True
        for k, v in behavior_obj.items():
            try:
                ik = int(k)
            except Exception:
                convertible = False
                break
            m[ik] = v
        if convertible and m:
            return m

    raise ValueError("behavior JSON 格式无法识别：请提供 list[trial_dict]、单个 trial_dict（含 'trial_index'），或 {trial_index: trial_dict}。")


# -------------------- 主类：封装交互逻辑（含 events 规则） --------------------
class GraphGazeViewer:
    def __init__(
        self,
        eye_tracking_df: pd.DataFrame,
        trial_map: dict,
        default_node_positions=None,
        AOI_r: float = 60.0,
        screen_size=(1920, 1080),
    ):
        """
        eye_tracking_df: 含列 ['X','Y','Time','trial_index','event'] 的眼动数据
        trial_map: {trial_index: trial_dict}，trial_dict 可包含 'graph','rewards','node_positions','events'
        default_node_positions: 当某个 trial 缺少 node_positions 时的默认坐标列表
        AOI_r: AOI 可视化半径（仅用于画圈）
        screen_size: 屏幕坐标范围 (width, height)
        """
        self.df = self._clean_eye_df(eye_tracking_df)
        self.trial_map = trial_map
        self.default_node_positions = default_node_positions or [
            (960, 162), (755, 222), (616, 382), (585, 593), (674, 787),
            (853, 902), (1066, 902), (1245, 787), (1334, 593), (1303, 382), (1164, 222),
        ]
        self.AOI_r = AOI_r
        self.screen_w, self.screen_h = screen_size

        # 状态容器
        self.current_trial = None
        self.X = self.Y = self.T = self.E = None
        self._static_artists = []
        self.node_art = {}   # node_id -> {'scatter':..., 'text':..., 'circle':...}
        self.edge_art = {}   # (i,j) -> FancyArrowPatch
        self.trial_events = {}  # trial_id -> list of events (sorted by 'time')

        # 交互控件
        self.trial_values = sorted(self.df["trial_index"].unique().tolist())
        self.trial_dropdown = widgets.Dropdown(options=self.trial_values, value=self.trial_values[0], description="trial_index")
        self.slider = widgets.IntSlider(value=0, min=0, max=1, step=1, description="t idx", continuous_update=True)
        self.play = widgets.Play(interval=0.02, value=0, min=0, max=1, step=1, disabled=False)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        self.ui = widgets.HBox([self.trial_dropdown, self.play, self.slider])

        # 画布
        self.fig, self.ax = plt.subplots(figsize=(8, 4.8))  # 16:9
        self.ax.set_xlim(0, self.screen_w)
        # self.ax.set_ylim(0, self.screen_h)
        self.ax.set_ylim(self.screen_h, 0)
        self.ax.set_xlabel("X (px)")
        self.ax.set_ylabel("Y (px)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.invert_yaxis()  # 屏幕坐标 y 向下
        plt.tight_layout()

        # 注视点与文字
        (self.dot,) = self.ax.plot([], [], "o", ms=8, color="red", zorder=5)
        self.event_text = self.ax.text(0.98, 0.04, "", ha="right", va="bottom", transform=self.ax.transAxes, fontsize=11)
        self.time_text  = self.ax.text(0.02, 0.04, "", ha="left",  va="bottom", transform=self.ax.transAxes, fontsize=11)

        # 绑定回调
        self.trial_dropdown.observe(self._on_trial_change, names="value")
        self.slider.observe(self._on_slider_change, names="value")

    @staticmethod
    def _clean_eye_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        assert {"X", "Y", "Time", "trial_index"}.issubset(df.columns), "CSV must contain X, Y, Time, trial_index."
        # 如无 'event' 列，填空字符串以避免后续索引报错
        if "event" not in df.columns:
            df["event"] = ""
        df = (
            df.dropna(subset=["X", "Y", "Time", "trial_index"])
              .astype({"X": float, "Y": float, "Time": float, "trial_index": int})
        )
        return df

    # ---------- 静态层：构建/清理 ----------
    def _clear_static(self):
        for art in self._static_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._static_artists = []
        self.node_art.clear()
        self.edge_art.clear()

    def _draw_nodes_and_graph(self, trial_id: int):
        tdata = self.trial_map.get(int(trial_id), {})

        node_positions = tdata.get("node_positions", self.default_node_positions)
        graph = tdata.get("graph", [[] for _ in range(len(node_positions))])
        rewards = tdata.get("rewards", [None] * len(node_positions))

        # 记录 events（按时间排序）
        events = tdata.get("events", [])
        if isinstance(events, list):
            events = sorted(events, key=lambda e: e.get("time", -np.inf))
        self.trial_events[trial_id] = events

        # 节点 + 奖励文本 + AOI 圈（初始隐藏，等 show graph 后再显示）
        for idx, (x, y) in enumerate(node_positions):
            sc = self.ax.scatter(x, y, color='gray', s=500, zorder=2, alpha=1.0, visible=False)
            txt_label = "NA" if (idx >= len(rewards) or rewards[idx] is None) else str(rewards[idx])
            txt = self.ax.text(x, y, txt_label, ha='center', va='center', color='white',
                            fontsize=10, zorder=3, visible=False)
            circ = Circle((x, y), self.AOI_r, fill=False, linestyle='--', linewidth=1,
                        alpha=0.5, zorder=1, visible=False)
            self.ax.add_patch(circ)

            self._static_artists.extend([sc, txt, circ])
            self.node_art[idx] = {"scatter": sc, "text": txt, "circle": circ, "pos": (x, y)}

        # 有向边（箭头）
        # 提升 mutation_scale（箭头更大），设置 shrinkA/B（避免箭头头部被节点盖住），
        # zorder=1（在圆圈之上、节点之下）
        for i, outs in enumerate(graph):
            if not outs:
                continue
            xi, yi = node_positions[i]
            for j in outs:
                if j is None or j < 0 or j >= len(node_positions):
                    continue
                xj, yj = node_positions[j]
                arrow = FancyArrowPatch(
                    (xi, yi), (xj, yj),
                    arrowstyle='-|>',
                    mutation_scale=22,      # ← 放大箭头
                    shrinkA=18, shrinkB=18, # ← 远离节点中心，避免被点盖住
                    linewidth=1.6, color='gray', alpha=0.9,
                    zorder=1,               # ← 比圆圈高、比节点低
                    connectionstyle="arc3",
                    visible=False
                )
                self.ax.add_patch(arrow)
                self._static_artists.append(arrow)
                self.edge_art[(i, j)] = arrow

    def _set_graph_visible(self, visible: bool):
        """显示/隐藏整个图（节点/圆/文字/箭头）。"""
        for d in self.node_art.values():
            d["scatter"].set_visible(visible)
            d["text"].set_visible(visible)
            d["circle"].set_visible(visible)
        for arrow in self.edge_art.values():
            arrow.set_visible(visible)

    # ---------- 动态层：应用事件规则 ----------
    def _apply_events_up_to(self, trial_id: int, current_time: float):
        """
        将 behavior events 发生到 current_time 的效果应用到图上。
        - 任何时刻仅保留“最后一次 get move / switch”对应的一条橙色箭头
        - show graph 之后才显示整图
        - select 时淡化 selected 节点（支持单个或列表），而不是淡化“当前所在节点”
        """
        # 可见性与状态
        graph_visible = False
        blue_node = None
        faded_nodes = set()
        last_orange_edge = None  # 只保留最后一条

        # 顺序应用（只到 current_time）
        for ev in self.trial_events.get(trial_id, []):
            if ev.get("time", -np.inf) > current_time:
                break

            etype = ev.get("event", "")
            if etype == "show graph":
                graph_visible = True

            elif etype == "visit":
                state = ev.get("state", None)
                if isinstance(state, int):
                    blue_node = state  # 当前蓝色节点

            elif etype == "get move":
                selected = ev.get("selected", None)
                if isinstance(selected, int) and isinstance(blue_node, int):
                    last_orange_edge = (blue_node, selected)  # 覆盖旧的

            elif etype == "switch":
                selected = ev.get("selected", None)
                if isinstance(selected, int) and isinstance(blue_node, int):
                    last_orange_edge = (blue_node, selected)  # 覆盖旧的

            elif etype == "select":
                selected = ev.get("selected", None)
                # 支持 selected 为单个或列表/元组
                if isinstance(selected, (list, tuple, set)):
                    for s in selected:
                        if isinstance(s, int):
                            faded_nodes.add(s)
                elif isinstance(selected, int):
                    faded_nodes.add(selected)

            elif etype == "done":
                graph_visible = False  # 整张图隐藏

        # 1) 显示/隐藏整图
        self._set_graph_visible(graph_visible)

        # 2) 节点：先还原→淡化 selected → 覆盖蓝色节点
        for nid, d in self.node_art.items():
            sc = d["scatter"]
            sc.set_facecolor('gray')
            sc.set_alpha(1.0)

        # 只淡化 selected 节点；若 selected 与 blue_node 相同，蓝色覆盖会在后一步生效
        for nid in faded_nodes:
            if nid in self.node_art and nid != blue_node:
                self.node_art[nid]["scatter"].set_alpha(0.3)

        # 蓝色节点覆盖
        if isinstance(blue_node, int) and blue_node in self.node_art:
            self.node_art[blue_node]["scatter"].set_facecolor('blue')
            self.node_art[blue_node]["scatter"].set_alpha(1.0)

        # 3) 箭头：全部还原为灰色，仅最后一条设为橙色
        for key, arrow in self.edge_art.items():
            arrow.set_color('gray')
            arrow.set_alpha(0.9)
            arrow.set_linewidth(1.6)
        if last_orange_edge is not None:
            arrow = self.edge_art.get(last_orange_edge)
            if arrow is not None:
                arrow.set_color('orange')
                arrow.set_alpha(1.0)
                arrow.set_linewidth(2.2)

    # ---------- 绑定/更新 ----------
    def _bind_trial(self, trial_id: int):
        self.current_trial = int(trial_id)
        d = (self.df[self.df["trial_index"] == trial_id]
             .sort_values("Time")
             .reset_index(drop=True))
        if d.empty:
            raise ValueError(f"No rows found for trial_index={trial_id}")

        self.X = d["X"].to_numpy(float)
        self.Y = d["Y"].to_numpy(float)
        self.T = d["Time"].to_numpy(float)
        self.E = d["event"].astype(str).to_numpy()

        self._clear_static()
        self._draw_nodes_and_graph(trial_id)

        # slider / play 重置
        self.slider.min = 0
        self.slider.max = len(d) - 1
        self.slider.step = 1
        self.slider.value = 0

        self.play.min = 0
        self.play.max = len(d) - 1
        self.play.step = 1
        self.play.value = 0

        self.ax.set_title(f"Trial {trial_id} — Fixation over Time")
        self.dot.set_data([], [])
        self.event_text.set_text("")
        self.time_text.set_text("")
        self.fig.canvas.draw_idle()

        # 初始时根据第一帧的时间应用一次事件（保证“show graph 之后才显示”）
        self._apply_events_up_to(self.current_trial, self.T[0])

    def _update(self, i: int):
        if self.X is None or len(self.X) == 0:
            return
        i = int(np.clip(i, 0, len(self.X) - 1))

        # 红点 & 文本
        self.dot.set_data([self.X[i]], [self.Y[i]])  # 传入序列避免报错
        self.event_text.set_text(f"event: {self.E[i]}")
        self.time_text.set_text(f"Time: {self.T[i]:.3f}")

        # 应用“到当前时间”的事件规则
        if self.current_trial is not None:
            self._apply_events_up_to(self.current_trial, self.T[i])

        self.fig.canvas.draw_idle()

    # ---- 回调 ----
    def _on_trial_change(self, change):
        if change["name"] == "value":
            self._bind_trial(change["new"])

    def _on_slider_change(self, change):
        if change["name"] == "value":
            self._update(change["new"])

    # ---- 对外接口 ----
    def show(self):
        """初始化并显示交互控件与图。"""
        self._bind_trial(self.trial_dropdown.value)
        display(self.ui)
        plt.show()


# -------------------- 便捷函数：一行创建并展示 --------------------
def interactive_graph_gaze_viewer(
    eye_tracking_df: pd.DataFrame,
    behavior_obj,
    default_node_positions=None,
    AOI_r: float = 60.0,
    screen_size=(1920, 1080),
):
    """
    读取/清洗眼动 df，构建 trial_map，并启动交互可视化（含 events 规则）。
    behavior_obj 可以是：
      - list[trial_dict]
      - 单个 trial_dict
      - dict[trial_id -> trial_dict]
      - dict 含 "trials": list[trial_dict]
    """
    trial_map = build_trial_map(behavior_obj)
    viewer = GraphGazeViewer(
        eye_tracking_df=eye_tracking_df,
        trial_map=trial_map,
        default_node_positions=default_node_positions,
        AOI_r=AOI_r,
        screen_size=screen_size,
    )
    viewer.show()
    return viewer  # 返回对象，便于后续访问 viewer.trial_dropdown 等控件