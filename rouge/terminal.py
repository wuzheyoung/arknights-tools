import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import json
import threading
import os
import copy
import math
from collections import deque
from heapq import heappush, heappop
from itertools import count
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageGrab
import unicodedata


# 未知的诡秘展开后的候选子类型: name -> (type, 每层数量上限, 是否必然出现)
# 上限 None 表示不限；必然出现的子类型在剩余名额不足时会被强制分配
GUIMI_SUBTYPES = {
    "不期而遇": ("event", None, False),
    "误入奇境": ("event", 1, False),
    "失与得": ("event", 1, False),
    "先行一步": ("event", 1, False),
    "秘境行商": ("shop", 1, True),
    "险路小径": ("exit", 1, False),
    "诡意行商": ("shop", 2, False),
    "狭路相逢": ("event", 2, False),
}

EXIT_NAMES = {"险路恶敌", "险路小径", "险路尽头"}

# 未知的凶戾展开后的候选子类型，紧急作战每层至少一个、上不封顶
XIONGLI_SUBTYPES = {
    "作战": ("combat", None),
    "紧急作战": ("combat", None),
}

UNKNOWN_NODE_KINDS = {
    "未知的诡秘": GUIMI_SUBTYPES,
    "未知的凶戾": XIONGLI_SUBTYPES,
}

# 无上限子类型参与剩余名额分配时使用的基准名额
UNLIMITED_QUOTA = 1


def count_known_subtypes(nodes):
    counts = {}
    for node in nodes:
        if node.get('type') == 'unknown':
            continue
        name = node.get('name')
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def get_guimi_quotas(known_counts):
    quotas = {}
    for name, (_, limit, _) in GUIMI_SUBTYPES.items():
        remaining = UNLIMITED_QUOTA if limit is None else limit - known_counts.get(name, 0)
        if remaining > 0:
            quotas[name] = remaining
    return quotas


def get_pending_mandatory(known_counts):
    # 必然出现但本层尚未露面的子类型及其缺口数量
    pending = {}
    for name, (_, limit, mandatory) in GUIMI_SUBTYPES.items():
        if not mandatory:
            continue
        need = (limit or 1) - known_counts.get(name, 0)
        if need > 0:
            pending[name] = need
    return pending


def get_guimi_candidates(known_counts, remaining_guimi):
    # 剩余未知诡秘刚好只够填必然节点时，只能是这些必然节点；否则按最坏打算取全部候选
    pending = get_pending_mandatory(known_counts)
    if pending and remaining_guimi <= sum(pending.values()):
        return list(pending)
    quotas = get_guimi_quotas(known_counts)
    return list(quotas)


def get_xiongli_candidates(known_counts, remaining_xiongli):
    if known_counts.get('紧急作战', 0) <= 0 and remaining_xiongli <= 1:
        return ['紧急作战']
    return ['作战', '紧急作战']


def predict_guimi_node(known_counts, consumed=None):
    quotas = get_guimi_quotas(known_counts)
    total = sum(quotas.values())
    if total <= 0:
        return []
    return [
        {"name": name, "type": GUIMI_SUBTYPES[name][0], "probability": quota / total}
        for name, quota in quotas.items()
    ]


def predict_xiongli_node(known_counts, unknown_xiongli_count, consumed=None):
    # 紧急作战每层至少一个：本层尚未出现时，剩余 N 个未知凶戾中恰好一个是紧急作战
    if known_counts.get('紧急作战', 0) > 0 or unknown_xiongli_count <= 0:
        emergency = 0.5
    else:
        emergency = 1.0 / unknown_xiongli_count
    return [
        {"name": "紧急作战", "type": "combat", "probability": emergency},
        {"name": "作战", "type": "combat", "probability": 1.0 - emergency},
    ]


def compute_unknown_predictions(nodes):
    known_counts = count_known_subtypes(nodes)
    unknown_nodes = [node for node in nodes if node.get('type') == 'unknown']
    xiongli_count = sum(1 for node in unknown_nodes if node.get('name') == '未知的凶戾')

    predictions_by_id = {}
    for node in unknown_nodes:
        kind = node.get('name')
        if kind not in UNKNOWN_NODE_KINDS:
            continue
        if kind == '未知的凶戾':
            predictions = predict_xiongli_node(known_counts, xiongli_count)
        else:
            predictions = predict_guimi_node(known_counts)
        if predictions:
            predictions_by_id[node['id']] = predictions
    return predictions_by_id


def get_shortest_distances(graph, start_id):
    distances = {start_id: 0}
    queue = deque([start_id])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, ()):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def get_distances_to_targets(graph, target_ids):
    distances = {}
    queue = deque((target_id, 0) for target_id in target_ids)
    while queue:
        current, distance = queue.popleft()
        if current in distances:
            continue
        distances[current] = distance
        queue.extend((neighbor, distance + 1) for neighbor in graph.get(current, ()))
    return distances


def enrich_unknown_predictions(nodes):
    predictions_by_id = compute_unknown_predictions(nodes)
    for node in nodes:
        predictions = predictions_by_id.get(node['id'])
        if predictions:
            node['prediction'] = predictions


# ==========================================
# 模块一：战术推演核心算法 (A* 寻路引擎)
# ==========================================
def get_direction_str(cx, cy, nx, ny):
    dx = nx - cx
    dy = ny - cy
    if dx == 0 and dy == 0: return "原地"
    dir_str = ""
    if dy > 0:
        dir_str += "上"
    elif dy < 0:
        dir_str += "下"
    if dx < 0:
        dir_str += "左"
    elif dx > 0:
        dir_str += "右"
    if abs(dx) > 1 or abs(dy) > 1: return f"{dir_str}(跨度 x:{dx} y:{dy})"
    return dir_str


def parts_to_tuple(parts_dict):
    return tuple(sorted((k, v) for k, v in parts_dict.items() if v > 0))


PART_COST_GAIN = {
    "报废轮子": (1, 0),
    "报废假肢": (1, 0),
    "标准引擎": (1, 0),
    "重弹簧": (0, 0),
    "气垫底座": (1, 0),
    "试作外骨骼": (1, 0),
    "一次性喷气背包": (0, 0),
    "老妈妈的融雪": (1, 0),
    "坎诺特的触须": (1, 0),
    "结构性原理": (1, 0),
    "简易遥控器": (0, 3),
}

# 单个首访节点能拿到的最高加分（calculate_node_value 中所有加分项的安全上界，
# 用于乐观上界剪枝，必须不低于任何实际可能取到的值，否则会误剪出最优解）
MAX_SINGLE_NODE_VALUE = 15.0


def is_combat_node(info):
    return (info.get('type') == 'combat' or info.get('name') == '险路恶敌') \
           and info.get('name') != '狭路相逢'


def get_node_outcomes(node_id, node_info, is_new, known_counts, consumed, unknown_totals):
    plain = [{"name": node_info.get('name', node_id),
              "type": node_info.get('type', ''),
              "probability": 1.0,
              "unknown_kind": None,
              "candidates": None}]
    if not is_new or node_info.get('type') != 'unknown':
        return plain

    kind = node_info.get('name')
    if kind not in UNKNOWN_NODE_KINDS:
        return plain
    # 本节点尚未计入 consumed，故剩余数包含它自己
    remaining = unknown_totals.get(kind, 0) - consumed.get(kind, 0)
    if kind == '未知的凶戾':
        candidates = get_xiongli_candidates(known_counts, remaining)
    else:
        candidates = get_guimi_candidates(known_counts, remaining)
    if not candidates:
        return plain
    # 不按最优子类型计分，改为按最坏打算取候选中的最低价值（见 calculate_node_value）
    return [dict(plain[0], unknown_kind=kind, candidates=candidates)]


def calculate_node_value(node_info, is_new, nodes_info, current_parts,
                         prefer_combat, avoid_combat, explore_map,
                         current_remaining_ap=0, visited_set=frozenset(), outcome=None,
                         settling=False):
    if not is_new: return 0

    candidates = outcome.get('candidates') if outcome else None
    if candidates:
        subtype_kinds = UNKNOWN_NODE_KINDS.get(outcome.get('unknown_kind'), {})
        base_candidate_val = min(
            calculate_node_value(
                node_info, is_new, nodes_info, current_parts,
                prefer_combat, avoid_combat, explore_map,
                current_remaining_ap=current_remaining_ap,
                visited_set=visited_set,
                outcome={"name": name, "type": subtype_kinds.get(name, ('event',))[0]},
                settling=settling)
            for name in candidates
        )

        # 补回被 min 递归提前 return 所吞噬的母节点探索加分
        if explore_map and "未知" in node_info.get('name', ''):
            base_candidate_val += 1.2

        return base_candidate_val

    base_val = 0
    scored_info = outcome or node_info
    n_name = scored_info.get('name', '')
    n_type = scored_info.get('type', '')

    # 出口只在选择结算时才计分，路过不给首访加分
    if n_name in EXIT_NAMES and not settling:
        return 0

    is_narrow_encounter = n_name == "狭路相逢"
    is_emergency_combat = n_name == "紧急作战" or n_type == 'emergency_combat'
    is_regular_combat = is_combat_node(scored_info) and not is_narrow_encounter

    if is_emergency_combat:
        base_val += 1.5
    elif is_regular_combat:
        base_val += 1.0
    elif n_name == "不期而遇":
        base_val += 1.3
    elif is_narrow_encounter:
        base_val += 1.5
        if prefer_combat:
            base_val += 0.8
    elif node_info.get('is_effective', False):
        base_val += 1.0

    if prefer_combat and is_regular_combat:
        base_val += 0.5
    elif avoid_combat and is_regular_combat:
        base_val -= 0.5
    if explore_map and ("未知" in n_name or n_type == 'vision'): base_val += 1.2
    if n_type == 'vision': base_val += 1.5

    if n_name == "险路小径":
        base_val += 2.5 if current_remaining_ap >= 2 else 0.3
    if n_name == "误入奇境":
        base_val += 4.0 if (sum(current_parts.values()) if current_parts else 0) > 0 else 0.1
    if n_name == "失与得": base_val += 1.0
    if current_remaining_ap <= 2:
        if "诡秘" in n_name or n_name == "诡意行商":
            has_visited_mijing = any(
                "秘境行商" in nodes_info[v].get('name', '') or nodes_info[v].get('type') == 'shop' for v in
                visited_set)
            base_val += 8.0 if has_visited_mijing else -5.0
    return base_val


def get_part_targets(part_name, curr, curr_x, curr_y, nodes_info, part_targets_cache):
    cache_key = (curr, part_name)
    if cache_key in part_targets_cache:
        return part_targets_cache[cache_key], PART_COST_GAIN[part_name]
    cost, ap_gain = PART_COST_GAIN[part_name]
    if part_name == "报废轮子":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and (info['x'] == curr_x or info['y'] == curr_y) and (
                               abs(info['x'] - curr_x) + abs(info['y'] - curr_y) <= 2)]
    elif part_name in ["报废假肢", "标准引擎"]:
        targets = [n for n, info in nodes_info.items() if
                   n != curr and max(abs(info['x'] - curr_x), abs(info['y'] - curr_y)) <= 1]
    elif part_name == "重弹簧":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and (abs(info['x'] - curr_x) + abs(info['y'] - curr_y) <= 2)]
    elif part_name == "气垫底座":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and (abs(info['x'] - curr_x) + abs(info['y'] - curr_y) <= 2)]
    elif part_name == "试作外骨骼":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and (info['x'] == curr_x or info['y'] == curr_y) and (
                               abs(info['x'] - curr_x) + abs(info['y'] - curr_y) <= 3)]
    elif part_name == "一次性喷气背包":
        targets = [n for n, info in nodes_info.items() if n != curr]
    elif part_name == "老妈妈的融雪":
        targets = [n for n, info in nodes_info.items() if n != curr and not is_combat_node(info)]
    elif part_name == "坎诺特的触须":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and (info.get('type') == 'shop' or info.get('name') == '秘境行商')]
    elif part_name in ["结构性原理"]:
        targets = [n for n, info in nodes_info.items() if n != curr]
    elif part_name == "简易遥控器":
        targets = [n for n, info in nodes_info.items() if
                   n != curr and max(abs(info['x'] - curr_x), abs(info['y'] - curr_y)) <= 1]
    else:
        targets = []
    part_targets_cache[cache_key] = targets
    return targets, (cost, ap_gain)


def solve_tree_sea_path(data, prefer_combat, avoid_combat, explore_map, layer=None):
    try:
        nodes_info = {n['id']: n for n in data['map_topology']['nodes']}
        edges = data['map_topology']['edges']
        start_id = data['player_state']['start_node_id']
        initial_ap = data['player_state']['current_ap']
        initial_parts = {p['name']: p['remaining_uses'] for p in data['player_state']['active_parts']}
        required_nodes = data['player_state'].get('required_nodes', [])
        layer = layer if layer is not None else data['player_state'].get('layer')
        try:
            layer = int(layer) if layer is not None else None
        except (TypeError, ValueError):
            layer = None
        # 真实出口识别：只有走到这里才会停止探索
        real_exits = {
            n_id for n_id, info in nodes_info.items()
            if info.get('type') == 'exit'
               or info.get('name') in ['险路恶敌', '险路小径', '险路尽头']
               or info.get('fixed_type') == 'exit'
        }
    except KeyError as e:
        return None, -1, -1, f"数据解析错误：地图数据缺少必要字段 {e}。"

    if start_id not in nodes_info:
        return None, -1, -1, f"错误：指定的起点坐标 {start_id} 不在当前地图数据中，请核对。"

    for req_node in required_nodes:
        if req_node not in nodes_info:
            return None, -1, -1, f"错误：指定的必经节点 {req_node} 不在当前地图数据中，请核对坐标。"

    graph = {n_id: [] for n_id in nodes_info}
    for u, v in edges:
        if u in graph and v in graph:
            graph[u].append(v)
            graph[v].append(u)

    distances = get_shortest_distances(graph, start_id)
    exit_distances = get_distances_to_targets(graph, real_exits)
    unknown_predictions = compute_unknown_predictions(list(nodes_info.values()))
    for node_id, predictions in unknown_predictions.items():
        nodes_info[node_id]['prediction'] = predictions

    known_counts = count_known_subtypes(list(nodes_info.values()))
    unknown_totals = {}
    for info in nodes_info.values():
        kind = info.get('name')
        if info.get('type') == 'unknown' and kind in UNKNOWN_NODE_KINDS:
            unknown_totals[kind] = unknown_totals.get(kind, 0) + 1

    start_name = nodes_info[start_id].get('name', str(start_id))

    tie_breaker = count()
    initial_queue_item = (
        0, -initial_ap, 0, next(tie_breaker), start_id, initial_ap,
        frozenset([start_id]), [f"当前位置: [{start_name}] (坐标:{start_id})"], initial_parts, {}
    )

    pq = [initial_queue_item]
    visited_states = {}
    best_path, best_score, best_remaining_ap = None, -999999, -1
    max_iterations, iterations = 500000, 0
    part_targets_cache = {}

    # 第五、六层允许在任意节点结束；其他层有出口时必须走到出口
    has_exit = len(real_exits) > 0
    require_exit = has_exit and layer not in (5, 6)

    while pq and iterations < max_iterations:
        iterations += 1
        neg_score, neg_ap, steps, _, curr, ap, visited, path, parts, consumed = heappop(pq)
        score = -neg_score

        # 站在出口节点时可选择结算：结算价值此刻才计入，路过则不计分继续行动
        at_exit = curr in real_exits
        settle_bonus = 0
        if at_exit:
            settle_bonus = calculate_node_value(
                nodes_info[curr], True, nodes_info, parts,
                prefer_combat, avoid_combat, explore_map,
                current_remaining_ap=ap, visited_set=visited,
                outcome=None, settling=True)
        reached_end = at_exit if require_exit else True
        if reached_end and all(req in visited for req in required_nodes):
            settled_score = score + settle_bonus
            if settled_score > best_score or (settled_score == best_score and ap > best_remaining_ap):
                best_score = settled_score
                best_remaining_ap = ap
                best_path = path + [f"在 [{nodes_info[curr].get('name', curr)}] 选择结算离开本层"] if at_exit else path

        # 出口可选择不结算，继续在本层行动，因此不再直接剪掉该分支

        state_key = (curr, parts_to_tuple(parts), visited, parts_to_tuple(consumed))
        pareto_points = visited_states.get(state_key)
        if pareto_points is not None:
            dominated = False
            for old_score, old_ap in pareto_points:
                if old_score >= score and old_ap >= ap:
                    dominated = True
                    break
            if dominated:
                continue
            pareto_points[:] = [
                (old_score, old_ap) for old_score, old_ap in pareto_points
                if not (score >= old_score and ap >= old_ap)
            ]
            pareto_points.append((score, ap))
        else:
            visited_states[state_key] = [(score, ap)]

        # 乐观上界剪枝：剩余未访问节点即使全部拿到单节点最高分，也追不上已知最优解时，
        # 该分支不可能产生更优结果，直接跳过（不改变最终解的正确性）
        unvisited_count = len(nodes_info) - len(visited)
        if score + MAX_SINGLE_NODE_VALUE * unvisited_count <= best_score:
            continue

        curr_x, curr_y = nodes_info[curr].get('x', 0), nodes_info[curr].get('y', 0)

        # 1. 徒步
        if ap >= 1:
            for nxt in graph[curr]:
                nxt_info = nodes_info[nxt]
                is_new = (nxt not in visited)
                ap_recovery = 1 if (is_new and nxt_info.get('type') == 'vision') else 0
                new_ap = ap - 1 + ap_recovery

                new_visited = visited | frozenset([nxt])
                direction = get_direction_str(curr_x, curr_y, nxt_info.get('x', 0), nxt_info.get('y', 0))

                ap_desc = "AP-1"
                if ap_recovery > 0: ap_desc += ", AP+1(羽瞰)"
                for outcome in get_node_outcomes(nxt, nxt_info, is_new, known_counts, consumed, unknown_totals):
                    new_score = score + calculate_node_value(
                        nxt_info, is_new, nodes_info, parts,
                        prefer_combat, avoid_combat, explore_map,
                        current_remaining_ap=new_ap,
                        visited_set=new_visited, outcome=outcome)
                    kind = outcome.get('unknown_kind')
                    new_consumed = consumed
                    if kind:
                        new_consumed = dict(consumed)
                        new_consumed[kind] = new_consumed.get(kind, 0) + 1
                    outcome_name = outcome['name'] if is_new and nxt_info.get('type') == 'unknown' else nxt_info.get('name', nxt)
                    new_path = path + [f"向【{direction}】徒步 -> [{outcome_name}] ({ap_desc})"]
                    heappush(pq, (
                        -new_score, -new_ap, steps + 1, next(tie_breaker), nxt, new_ap, new_visited,
                        new_path, parts, new_consumed))

        # 2. 零件使用
        for part_name, uses in {k: v for k, v in parts.items() if v > 0}.items():
            new_parts = parts.copy()
            new_parts[part_name] -= 1
            targets, (cost, ap_gain) = get_part_targets(part_name, curr, curr_x, curr_y, nodes_info, part_targets_cache)

            if ap >= cost:
                for nxt in targets:
                    nxt_info = nodes_info[nxt]
                    is_new = (nxt not in visited)

                    # 补全零件降落时的羽瞰点 AP 恢复判定
                    ap_recovery = 1 if (is_new and nxt_info.get('type') == 'vision') else 0
                    new_ap = ap - cost + ap_gain + ap_recovery

                    new_visited = visited | frozenset([nxt])
                    direction = get_direction_str(curr_x, curr_y, nxt_info.get('x', 0), nxt_info.get('y', 0))

                    # 补全日志描述中的收益细分
                    action_desc = f"AP-{cost}" if cost > 0 else "无消耗"
                    if ap_gain > 0: action_desc += f", AP+{ap_gain}(零件)"
                    if ap_recovery > 0: action_desc += f", AP+{ap_recovery}(羽瞰)"
                    for outcome in get_node_outcomes(nxt, nxt_info, is_new, known_counts, consumed, unknown_totals):
                        new_score = score + calculate_node_value(
                            nxt_info, is_new, nodes_info, new_parts,
                            prefer_combat, avoid_combat, explore_map,
                            current_remaining_ap=new_ap,
                            visited_set=new_visited, outcome=outcome)
                        kind = outcome.get('unknown_kind')
                        new_consumed = consumed
                        if kind:
                            new_consumed = dict(consumed)
                            new_consumed[kind] = new_consumed.get(kind, 0) + 1
                        outcome_name = outcome['name'] if is_new and nxt_info.get('type') == 'unknown' else nxt_info.get('name', nxt)
                        new_path = path + [
                            f"向【{direction}】使用零件[{part_name}] -> [{outcome_name}] ({action_desc})"]
                        heappush(pq, (
                            -new_score, -new_ap, steps + 1, next(tie_breaker), nxt, new_ap, new_visited,
                            new_path, new_parts, new_consumed))

    return best_path, best_score, best_remaining_ap, ""


# ==========================================
# 模块二：高精度 OCR 与拓扑匹配
# ==========================================
def get_visual_width(text):
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width


def pad_to_visual_width(text, target_width):
    padding = max(0, target_width - get_visual_width(text))
    return text + " " * padding


class DynamicTopologyRecognizer:
    def __init__(self, log_callback):
        self.log = log_callback
        self.reader = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.unified_json_path = os.path.join(current_dir, "Layer_Skeletons.json")
        self.easyocr_model_dir = os.path.join(current_dir, "easyocr_models")

        self.node_mapping = {
            "诡秘": ("未知的诡秘", "unknown", True), "凶戾": ("未知的凶戾", "unknown", True),
            "作战": ("作战", "combat", True), "紧急": ("紧急作战", "combat", True),
            "不期": ("不期而遇", "event", True), "秘境行商": ("秘境行商", "shop", True),
            "诡意行商": ("诡意行商", "shop", True), "先行": ("先行一步", "event", True),
            "羽瞰": ("羽瞰点", "vision", False), "密道": ("曲折密道", "teleport", False),
            "空地": ("林间空地", "empty", False), "恶敌": ("险路恶敌", "exit", True),
            "小径": ("险路小径", "exit", False), "尽头": ("险路尽头", "exit", False),
            "奇境": ("误入奇境", "event", True), "失与得": ("失与得", "event", True),
            "狭路相逢": ("狭路相逢", "event", True)
        }

    def init_engine(self):
        if self.reader is None:
            self.log(">>> [系统] 正在后台初始化 EasyOCR 引擎，请稍候...")
            if os.path.isdir(self.easyocr_model_dir):
                self.reader = easyocr.Reader(
                    ['ch_sim'], gpu=True,
                    model_storage_directory=self.easyocr_model_dir,
                    download_enabled=False)
            else:
                self.reader = easyocr.Reader(['ch_sim'], gpu=True)
            self.log(">>> [系统] OCR 引擎加载完毕，系统就绪！")

    def _read_layer_skeleton(self, target_layer):
        if not os.path.exists(self.unified_json_path):
            self.log(f"[错误] 找不到地图骨架总文件: {self.unified_json_path}")
            return []
        try:
            with open(self.unified_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 根据输入的层数，过滤出对应的 Skeleton
            prefix = f"Layer_{target_layer}-"
            candidates = [t for t in data if str(t.get('template_id', '')).startswith(prefix)]

            if not candidates:
                self.log(f"[警告] 文件中未找到前缀为 {prefix} 的层数数据。退回全量模板比对。")
                return data
            return candidates
        except Exception as e:
            self.log(f"[错误] 解析 {self.unified_json_path} 失败: {e}")
        return []

    def calculate_sparse_grid(self, coords, is_x=True, reverse=False):
        if not coords: return {}
        sorted_coords = sorted(list(set(coords)))
        clusters, current_cluster = [], [sorted_coords[0]]
        for val in sorted_coords[1:]:
            if val - current_cluster[-1] <= 45:
                current_cluster.append(val)
            else:
                clusters.append(current_cluster)
                current_cluster = [val]
        clusters.append(current_cluster)

        centers = [sum(c) / len(c) for c in clusters]
        if len(centers) == 1:
            return {val: 0 for val in clusters[0]}

        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        min_expected_step = 100 if is_x else 60
        valid_diffs = [d for d in diffs if d >= min_expected_step]
        base_step = min(valid_diffs) if valid_diffs else min_expected_step

        mapping = {}
        for c in clusters:
            center = sum(c) / len(c)
            idx = int(round((center - centers[0]) / base_step))
            if reverse: idx = -idx
            for val in c: mapping[val] = idx
        return mapping

    def process_to_dict(self, img_pil, layer_num, manual_start_id, current_ap=8, target_template=""):
        if self.reader is None:
            self.log("[错误] OCR引擎尚未加载，请稍候。")
            return None

        candidates = self._read_layer_skeleton(layer_num)
        if not candidates: return None

        if target_template:
            filtered = [c for c in candidates if target_template in str(c.get('template_id', ''))]
            if filtered:
                candidates = filtered
                self.log(f">>> [系统] 已指定匹配范围：限定模板 '{target_template}'")
            else:
                self.log(f">>> [警告] 未找到匹配 '{target_template}' 的模板，退回自动匹配。")

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        self.log(">>> 正在进行高精度 OCR 扫描，提取拓扑特征...")

        scale = 2.0
        scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        valid_chars = '未知的诡秘凶戾作战紧急不期而遇秘境诡意行商羽瞰点曲折密道林间空地险路恶敌小径尽头奇境失与得狭路相逢先一步'
        raw_results = self.reader.readtext(enhanced_gray, allowlist=valid_chars)

        detected_nodes_pixels = []
        for bbox, text, prob in raw_results:
            for key, (std_name, n_type, eff) in self.node_mapping.items():
                if key in text:
                    cx = int((bbox[0][0] + bbox[1][0]) / 2 / scale)
                    cy = int((bbox[0][1] + bbox[2][1]) / 2 / scale)
                    detected_nodes_pixels.append({
                        "pixel_x": cx, "pixel_y": cy,
                        "name": std_name, "type": n_type, "is_eff": eff
                    })
                    break

        if not detected_nodes_pixels:
            self.log("[错误] 未能在截图中识别到有效节点，请检查截图是否包含文字。")
            return None

        x_mapping = self.calculate_sparse_grid([n["pixel_x"] for n in detected_nodes_pixels], is_x=True)
        y_mapping = self.calculate_sparse_grid([n["pixel_y"] for n in detected_nodes_pixels], is_x=False, reverse=True)

        for node in detected_nodes_pixels:
            node["phys_x"] = x_mapping[node["pixel_x"]]
            node["phys_y"] = y_mapping[node["pixel_y"]]

        active_skeleton, best_score, best_match_score, offset_x, offset_y = None, -10 ** 9, 0, 0, 0
        EXIT_WEIGHT = 5  # 出口是最可靠地标，优先按 exit 位置对齐

        for skeleton in candidates:
            skel_coords = {(n["grid_x"], n["grid_y"]) for n in skeleton["nodes"]}
            skel_exits = {(n["grid_x"], n["grid_y"]) for n in skeleton["nodes"]
                          if n.get("fixed_type") == "exit"}
            for dn in detected_nodes_pixels:
                for sn in skeleton["nodes"]:
                    ox = sn["grid_x"] - dn["phys_x"]
                    oy = sn["grid_y"] - dn["phys_y"]
                    raw, score = 0, 0
                    for d in detected_nodes_pixels:
                        tgt = (d["phys_x"] + ox, d["phys_y"] + oy)
                        if tgt in skel_coords:
                            raw += 1
                            score += 1
                            if d["type"] == "exit":
                                # 检出出口命中骨架出口→重奖；落到非出口位置→重罚
                                score += EXIT_WEIGHT if tgt in skel_exits else -EXIT_WEIGHT
                        elif d["type"] == "exit":
                            score -= 1  # 检出的出口没落在骨架任何节点上，轻罚

                    if score > best_score:
                        best_score = score
                        best_match_score = raw
                        active_skeleton = skeleton
                        offset_x, offset_y = ox, oy

        if not active_skeleton:
            self.log("[错误] 无法与任何本地蓝图骨架对齐。")
            return None

        final_nodes = []
        final_edges = []
        aligned_detected_dict = {(n["phys_x"] + offset_x, n["phys_y"] + offset_y): n for n in detected_nodes_pixels}

        # 1. 节点处理：严格遍历骨架节点，绝不新增或删除节点[cite: 1]
        #    骨架信息优先级：fixed_name(骨架显式命名) > fixed_type(骨架锁定类型) > OCR 赋名
        for snode in active_skeleton["nodes"]:
            coord = (snode["grid_x"], snode["grid_y"])
            fixed_t = snode.get("fixed_type")
            skeleton_name = snode.get("fixed_name")  # 骨架自带的权威名称(如起点/险路尽头)

            if skeleton_name:
                # 【最高优先级】骨架 fixed_name 权威命名(起点等)，无视该位置的 OCR 结果[cite: 1]
                final_nodes.append({
                    "id": coord, "name": skeleton_name, "type": fixed_t,
                    "is_effective": True, "x": snode["grid_x"], "y": snode["grid_y"]
                })
            elif fixed_t:
                # 【骨架锁定类型/OCR赋名】如出口(exit)、战斗(combat)：类型由骨架锁定，
                # 名字优先取 OCR(区分险路恶敌/小径/尽头)，OCR缺失或空名时回退骨架默认名
                dn = aligned_detected_dict.get(coord)
                default_name = {"exit": "险路尽头", "combat": "作战"}.get(fixed_t, "未探明区域")
                ocr_name = dn["name"] if (dn and dn["name"]) else default_name
                final_nodes.append({
                    "id": coord, "name": ocr_name, "type": fixed_t,
                    "is_effective": dn["is_eff"] if dn else True,
                    "x": snode["grid_x"], "y": snode["grid_y"]
                })
            elif coord in aligned_detected_dict:
                # 【仅限赋名】骨架中存在且无固定属性，将 OCR 识别的名称赋给它；
                # 诡秘/凶戾为 unknown 型，其具体子类型稍后由排除法预测
                dn = aligned_detected_dict[coord]
                final_nodes.append({
                    "id": coord, "name": dn["name"], "type": dn["type"],
                    "is_effective": dn["is_eff"], "x": snode["grid_x"], "y": snode["grid_y"]
                })
            else:
                # 【空白兜底】骨架中有此节点，但 OCR 未识别到任何文字，视为无效果的林间空地
                final_nodes.append({
                    "id": coord, "name": "林间空地", "type": "empty",
                    "is_effective": False, "x": snode["grid_x"], "y": snode["grid_y"]
                })

        # 强大的 [x,y] 到 (x,y) 类型转换器[cite: 1]
        def convert_id(node_id):
            if isinstance(node_id, list) and len(node_id) == 2:
                return (int(node_id[0]), int(node_id[1]))
            if isinstance(node_id, str):
                node_id = node_id.strip()
                if node_id.startswith("(") and node_id.endswith(")"):
                    parts = node_id.strip("()").split(",")
                    if len(parts) == 2: return (int(parts[0].strip()), int(parts[1].strip()))
                if node_id.startswith("Node_"):
                    parts = node_id.split('_')
                    if len(parts) >= 3: return (int(parts[1]), int(parts[2]))
            return node_id

        # 2. 连线处理：完全沿用骨架设定的连线逻辑，不参考 OCR，不进行增删改[cite: 1]
        for u, v in active_skeleton["edges"]:
            final_edges.append([convert_id(u), convert_id(v)])

        self.log("\n" + "=" * 25 + " 【可视化地图拓扑】 " + "=" * 25)
        self.log(
            f" 📂 对齐模板: {active_skeleton.get('template_id', 'Unknown')} | 匹配度: {best_match_score}/{len(detected_nodes_pixels)}\n")

        name_abbrev_map = {
            "未知的诡秘": "诡秘", "未知的凶戾": "凶戾", "作战": "作战", "紧急作战": "紧急",
            "不期而遇": "不期", "秘境行商": "秘商", "诡意行商": "诡商", "先行一步": "先行",
            "羽瞰点": "羽瞰", "曲折密道": "密道",
            "林间空地": "空地", "险路恶敌": "恶敌", "险路小径": "小径", "险路尽头": "尽头",
            "误入奇境": "奇境", "失与得": "失得", "狭路相逢": "狭路", "起点": "起点"
        }

        if final_nodes:
            max_x = max([n['x'] for n in final_nodes])
            max_y = max([n['y'] for n in final_nodes])

            def has_edge(x1, y1, x2, y2):
                return any((e[0] == (x1, y1) and e[1] == (x2, y2)) or (e[1] == (x1, y1) and e[0] == (x2, y2)) for e in
                           final_edges)

            cell_width = 10
            header = "    "
            for x in range(max_x + 1): header += pad_to_visual_width(f"X{x}", cell_width)
            self.log(header)

            for y in range(max_y, -1, -1):
                row_str = f"Y{y:<2} "
                for x in range(max_x + 1):
                    node = next((n for n in final_nodes if n['x'] == x and n['y'] == y), None)
                    has_right = has_edge(x, y, x + 1, y)
                    conn = "——" if has_right else "  "

                    if node:
                        raw_name = node['name']
                        short_name = name_abbrev_map.get(raw_name, raw_name[:2])
                        cell_content = f"[{short_name}]{conn}"
                    else:
                        cell_content = ""
                    row_str += pad_to_visual_width(cell_content, cell_width)
                self.log(row_str)

                if y > 0:
                    conn_str = "    "
                    for x in range(max_x + 1):
                        has_down = has_edge(x, y, x, y - 1)
                        has_dr = has_edge(x, y, x + 1, y - 1)
                        has_ur = has_edge(x, y - 1, x + 1, y)
                        c_down = "│" if has_down else " "
                        c_diag = "  "
                        if has_dr and has_ur:
                            c_diag = " X"
                        elif has_dr:
                            c_diag = " \\"
                        elif has_ur:
                            c_diag = " /"
                        line_content = f"  {c_down}   {c_diag}"
                        conn_str += pad_to_visual_width(line_content, cell_width)
                    self.log(conn_str)
        self.log("=" * 68 + "\n")

        # 起点留空 -> 自动采用骨架固定起点(fixed_type=='start' 或名为'起点')
        resolved_start = manual_start_id
        if not resolved_start:
            start_node = next(
                (n for n in final_nodes if n.get('type') == 'start' or n.get('name') == '起点'), None)
            if start_node:
                resolved_start = start_node['id']
                self.log(f">>> [系统] 起点留空，自动采用骨架固定起点: {resolved_start}")
            else:
                self.log("[警告] 骨架中未找到固定起点，请在推演前手动输入起点坐标。")

        enrich_unknown_predictions(final_nodes)

        return {
            "map_topology": {"nodes": final_nodes, "edges": final_edges},
            "player_state": {
                "start_node_id": resolved_start,
                "current_ap": current_ap,
                "active_parts": [], "required_nodes": []
            }
        }


# ==========================================
# 模块三：统一 GUI 界面
# ==========================================
class UltimateTerminalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("黑流树海 - 全自动战术终端 V7.7 (全兼容排版版)")
        self.root.geometry("880x850")
        self.root.configure(padx=10, pady=10)

        self.loaded_image = None
        self.current_map_data = None

        def parse_node_id(raw_str):
            s = raw_str.strip().replace('，', ',')
            if ',' in s:
                parts = s.split(',')
                try:
                    return (int(parts[0].strip()), int(parts[1].strip()))
                except ValueError:
                    pass
            return s

        self.parse_node_id = parse_node_id

        # GUI 绘制部分（与之前相同，直接复用以保持兼容性）
        frame_image = tk.LabelFrame(root, text=" 加载游戏截图 ", padx=10, pady=8, font=("微软雅黑", 10, "bold"))
        frame_image.pack(fill=tk.X, pady=5)
        btn_frame1 = tk.Frame(frame_image)
        btn_frame1.pack(fill=tk.X)
        tk.Button(btn_frame1, text="📋 从剪贴板提取", command=self.load_from_clipboard, width=20).pack(side=tk.LEFT,
                                                                                                      padx=5)
        tk.Button(btn_frame1, text="📂 从本地文件选择", command=self.load_from_file, width=20).pack(side=tk.LEFT, padx=5)
        self.lbl_img_status = tk.Label(frame_image, text="⭕ 状态：未加载图像", fg="red", font=("微软雅黑", 10))
        self.lbl_img_status.pack(side=tk.LEFT, padx=20)

        frame_params = tk.LabelFrame(root, text=" 基础参数与识别 ", padx=10, pady=8, font=("微软雅黑", 10, "bold"))
        frame_params.pack(fill=tk.X, pady=5)
        param_frame = tk.Frame(frame_params)
        param_frame.pack(fill=tk.X, pady=5)

        # 第一行：层数、AP、起点
        tk.Label(param_frame, text="当前层数:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.entry_layer = tk.Entry(param_frame, width=8)
        self.entry_layer.grid(row=0, column=1, padx=5, sticky=tk.W)
        tk.Label(param_frame, text="当前 AP:").grid(row=0, column=2, padx=(20, 5), sticky=tk.W)
        self.entry_ap = tk.Entry(param_frame, width=8)
        self.entry_ap.grid(row=0, column=3, padx=5, sticky=tk.W)
        tk.Label(param_frame, text="起点坐标 (如 1,3):").grid(row=0, column=4, padx=(20, 5), sticky=tk.W)
        self.entry_start = tk.Entry(param_frame, width=12)
        self.entry_start.grid(row=0, column=5, padx=5, sticky=tk.W)

        # 第二行：加回指定模板功能
        tk.Label(param_frame, text="预设骨架 (选填):").grid(row=1, column=0, padx=5, pady=(5, 0), sticky=tk.W)
        self.entry_template = tk.Entry(param_frame, width=20)
        self.entry_template.grid(row=1, column=1, columnspan=3, padx=5, pady=(5, 0), sticky=tk.W)

        self.btn_ocr = tk.Button(frame_params, text="🔍 提取图像拓扑特征存入系统", bg="#2196F3", fg="white",
                                 font=("微软雅黑", 10, "bold"), command=self.run_ocr_thread)
        self.btn_ocr.pack(fill=tk.X, pady=(10, 0))

        frame_parts = tk.LabelFrame(root, text=" 持有加工品配置 (留空即为 0 次) ", padx=10, pady=10,
                                    font=("微软雅黑", 10, "bold"))
        frame_parts.pack(fill=tk.X, pady=5)
        parts_inner = tk.Frame(frame_parts)
        parts_inner.pack(fill=tk.X)

        def create_part_entry(parent, label_text, row, col):
            tk.Label(parent, text=label_text).grid(row=row, column=col, padx=(0, 2), pady=4, sticky=tk.E)
            entry = tk.Entry(parent, width=5)
            entry.grid(row=row, column=col + 1, padx=(0, 15))
            return entry

        self.entry_wheel = create_part_entry(parts_inner, "报废轮子:", 0, 0)
        self.entry_prosthetic = create_part_entry(parts_inner, "报废假肢:", 0, 2)
        self.entry_engine = create_part_entry(parts_inner, "标准引擎:", 0, 4)
        self.entry_spring = create_part_entry(parts_inner, "重弹簧:", 0, 6)
        self.entry_hover = create_part_entry(parts_inner, "气垫底座:", 1, 0)
        self.entry_exo = create_part_entry(parts_inner, "试作外骨骼:", 1, 2)
        self.entry_jetpack = create_part_entry(parts_inner, "喷气背包:", 1, 4)
        self.entry_snow = create_part_entry(parts_inner, "老妈妈的融雪:", 1, 6)
        self.entry_tentacle = create_part_entry(parts_inner, "坎诺特的触须:", 2, 0)
        self.entry_principle = create_part_entry(parts_inner, "结构性原理:", 2, 2)
        self.entry_remote = create_part_entry(parts_inner, "简易遥控器:", 2, 4)

        frame_pref = tk.LabelFrame(root, text=" 战术偏好与推演 ", padx=10, pady=8, font=("微软雅黑", 10, "bold"))
        frame_pref.pack(fill=tk.X, pady=5)
        pref_inner = tk.Frame(frame_pref)
        pref_inner.pack(fill=tk.X, pady=2)
        self.var_prefer_combat = tk.BooleanVar(value=False)
        self.var_avoid_combat = tk.BooleanVar(value=False)
        self.var_explore = tk.BooleanVar(value=True)
        tk.Checkbutton(pref_inner, text="⚔️ 倾向战斗", variable=self.var_prefer_combat,
                       command=self.on_prefer_combat_toggle).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(pref_inner, text="🛡️ 避免战斗", variable=self.var_avoid_combat,
                       command=self.on_avoid_combat_toggle).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(pref_inner, text="🌟 探索未知", variable=self.var_explore).pack(side=tk.LEFT, padx=5)
        tk.Label(pref_inner, text=" |  🚩 必经节点坐标:").pack(side=tk.LEFT, padx=(10, 5))
        self.entry_required = tk.Entry(pref_inner, width=20)
        self.entry_required.pack(side=tk.LEFT)
        self.btn_calc = tk.Button(frame_pref, text="⚡ 生成最佳战术路线 ⚡", font=("微软雅黑", 12, "bold"), bg="#4CAF50",
                                  fg="white", command=self.run_calc_thread)
        self.btn_calc.pack(fill=tk.X, pady=(10, 0))

        frame_log = tk.LabelFrame(root, text=" 终端日志 ", padx=10, pady=5)
        frame_log.pack(fill=tk.BOTH, expand=True, pady=0)
        self.text_output = scrolledtext.ScrolledText(frame_log, height=18, bg="#1E1E1E", fg="#00FF00",
                                                     font=("Consolas", 10))
        self.text_output.pack(fill=tk.BOTH, expand=True)

        self.ocr_engine = DynamicTopologyRecognizer(log_callback=self.safe_log)
        threading.Thread(target=self.ocr_engine.init_engine, daemon=True).start()

    def load_from_clipboard(self):
        try:
            img_data = ImageGrab.grabclipboard()
            if not img_data:
                self.safe_log("[警告] 剪贴板为空。")
                return
            self.loaded_image = Image.open(img_data[0]).convert("RGB") if isinstance(img_data,
                                                                                     list) else img_data.convert("RGB")
            self.lbl_img_status.config(text="✅ 状态：已读取剪贴板", fg="#4CAF50")
            self.safe_log(">>> [系统] 成功加载剪贴板图像。")
        except Exception as e:
            self.safe_log(f"[错误] 读取剪贴板失败: {e}")

    def load_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path: return
        self.loaded_image = Image.open(file_path).convert("RGB")
        self.lbl_img_status.config(text=f"✅ 状态：已加载 ({os.path.basename(file_path)})", fg="#4CAF50")
        self.safe_log(f">>> [系统] 加载图像: {file_path}")

    def on_prefer_combat_toggle(self):
        if self.var_prefer_combat.get(): self.var_avoid_combat.set(False)

    def on_avoid_combat_toggle(self):
        if self.var_avoid_combat.get(): self.var_prefer_combat.set(False)

    def safe_log(self, message):
        self.root.after(0, self._insert_log, message)

    def _insert_log(self, message):
        self.text_output.insert(tk.END, message + "\n")
        self.text_output.see(tk.END)

    def run_ocr_thread(self):
        if self.loaded_image is None:
            self.safe_log("[错误] 未加载图像。")
            return
        layer, ap, raw_start = self.entry_layer.get().strip(), self.entry_ap.get().strip(), self.entry_start.get().strip()
        target_tpl = self.entry_template.get().strip()  # 获取预设骨架名称

        if not layer.isdigit() or not ap.isdigit():
            self.safe_log("[错误] 层数和AP必须为纯数字。")
            return

        # 起点可留空：留空时自动采用骨架的固定起点，仅中途重规划才需手动输入
        start_id = self.parse_node_id(raw_start) if raw_start else None
        self.btn_ocr.config(state=tk.DISABLED)
        # 将 target_tpl 传给子线程
        threading.Thread(target=self._ocr_worker, args=(self.loaded_image.copy(), layer, start_id, int(ap), target_tpl),
                         daemon=True).start()

    def _ocr_worker(self, img_pil, layer, start_id, ap, target_tpl):
        # 接收并传递 target_template
        result_dict = self.ocr_engine.process_to_dict(img_pil, layer, manual_start_id=start_id, current_ap=ap,
                                                      target_template=target_tpl)
        if result_dict:
            self.current_map_data = result_dict
            self.safe_log(">>> [系统] 拓扑特征已存入内存。可以开始推演。")
        self.root.after(0, lambda: self.btn_ocr.config(state=tk.NORMAL))

    def _get_entry_count(self, widget):
        val = widget.get().strip()
        return int(val) if val.isdigit() else 0

    def run_calc_thread(self):
        if not self.current_map_data:
            self.safe_log("[拒绝操作] 内存中无地图数据！")
            return
        self.btn_calc.config(state=tk.DISABLED)
        threading.Thread(target=self._calc_worker, daemon=True).start()

    def _calc_worker(self):
        try:
            data = copy.deepcopy(self.current_map_data)
            ui_layer = self.entry_layer.get().strip()
            if ui_layer.isdigit(): data['player_state']['layer'] = int(ui_layer)
            ui_ap = self.entry_ap.get().strip()
            if ui_ap.isdigit(): data['player_state']['current_ap'] = int(ui_ap)
            raw_start = self.entry_start.get().strip()
            if raw_start: data['player_state']['start_node_id'] = self.parse_node_id(raw_start)
            req_str = self.entry_required.get().strip()
            if req_str: data['player_state']['required_nodes'] = [self.parse_node_id(n) for n in req_str.split('|') if
                                                                  n.strip()]

            active_parts = []
            parts_mapping = [('entry_wheel', '报废轮子'), ('entry_prosthetic', '报废假肢'),
                             ('entry_engine', '标准引擎'), ('entry_spring', '重弹簧'), ('entry_hover', '气垫底座'),
                             ('entry_exo', '试作外骨骼'), ('entry_jetpack', '一次性喷气背包'),
                             ('entry_snow', '老妈妈的融雪'), ('entry_tentacle', '坎诺特的触须'),
                             ('entry_principle', '结构性原理'), ('entry_remote', '简易遥控器')]
            for attr_name, part_name in parts_mapping:
                count = self._get_entry_count(getattr(self, attr_name, None))
                if count > 0: active_parts.append({"name": part_name, "remaining_uses": count})
            data['player_state']['active_parts'] = active_parts

            self.safe_log("\n" + "-" * 50)
            self.safe_log(
                f">>> 启动 A* 智能战术推演 (起点坐标: {data['player_state']['start_node_id']} / 初始AP: {data['player_state']['current_ap']}) ...")
            path, max_score, left_ap, err_msg = solve_tree_sea_path(data, self.var_prefer_combat.get(),
                                                                    self.var_avoid_combat.get(), self.var_explore.get(),
                                                                    layer=data['player_state'].get('layer'))

            if err_msg:
                self.safe_log(err_msg)
            elif path:
                self.safe_log("\n[√] 推演成功！最优路线如下：")
                for step in path: self.safe_log("    " + step)
                self.safe_log(f"======================================")
                self.safe_log(f"📊 收益得分 = {max_score:.1f} | 🔋 最终余留 AP = {left_ap}点")
                self.safe_log(f"======================================")
            else:
                self.safe_log("\n[X] 绝境警告：无法在当前 AP/必经条件下抵达终点(出口)。")
        except Exception as ex:
            self.safe_log(f"\n[严重错误] 算法异常: {str(ex)}")
        finally:
            self.root.after(0, lambda: self.btn_calc.config(state=tk.NORMAL))


if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateTerminalApp(root)
    root.mainloop()
