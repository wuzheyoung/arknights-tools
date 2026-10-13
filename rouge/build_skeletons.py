import re
import json
import os

# ==========================================
# 从 map.txt (SVG 拓扑预览) 生成 Layer_Skeletons.json
# 坐标系: SVG 的 y 轴向下, 游戏逻辑 y 轴向上 => grid_y = MAX_Y - cy
# ==========================================
LAYER = 1         # 当前这批模板所属层数
MAX_Y = 3         # viewBox 高度决定, y 范围 0~4 (5 行)
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(CUR_DIR, "map.txt")
OUT = os.path.join(CUR_DIR, "Layer_Skeletons.json")

BUTTON_RE = re.compile(r'<button class="blackstream-topology-card.*?</button>', re.S)
CIRCLE_RE = re.compile(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"[^>]*data-node-type="([^"]+)"')
LINE_RE = re.compile(r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"')
TITLE_RE = re.compile(r'title="([^"]+)"')
TITLE_START_RE = re.compile(r'起点\((\d+),(\d+)\)')
TITLE_EXIT_RE = re.compile(r'终点\((\d+),(\d+)\)')

# SVG 出口节点的可能类型名(不同层批次命名不一致)
EXIT_TYPES = {"danger-end", "danger-enemy", "exit"}


def to_grid(cx, cy):
    return int(round(float(cx))), MAX_Y - int(round(float(cy)))


def build_card(block, idx):
    title = TITLE_RE.search(block)
    title = title.group(1) if title else f"card_{idx}"

    # 标题里的坐标是游戏真实坐标, 作为起点/终点的权威校验基准(终点可能有多个)
    ts = TITLE_START_RE.search(title)
    title_start = (int(ts.group(1)), int(ts.group(2))) if ts else None
    title_exits = {(int(x), int(y)) for x, y in TITLE_EXIT_RE.findall(title)}

    nodes, edges, warnings = [], [], []
    coord_set = set()
    start_coord = None
    exit_coords = set()

    for cx, cy, ntype in CIRCLE_RE.findall(block):
        gx, gy = to_grid(cx, cy)
        coord_set.add((gx, gy))
        node = {"grid_x": gx, "grid_y": gy}
        if ntype == "start":
            node["fixed_name"] = "起点"
            node["fixed_type"] = "start"
            start_coord = (gx, gy)
        elif ntype in EXIT_TYPES:
            # 只固定 type，name 留给 OCR(区分险路恶敌/小径/尽头，供算法评分)
            node["fixed_type"] = "exit"
            exit_coords.add((gx, gy))
        elif ntype == "combat":
            node["fixed_type"] = "combat"
        nodes.append(node)

    # 保证起点和终点: 用标题坐标核对换算结果
    if title_start and start_coord != title_start:
        warnings.append(f"起点坐标 {start_coord} 与标题 {title_start} 不符 (检查 MAX_Y)")
    if title_exits and exit_coords != title_exits:
        warnings.append(f"终点坐标 {sorted(exit_coords)} 与标题 {sorted(title_exits)} 不符")

    for x1, y1, x2, y2 in LINE_RE.findall(block):
        a, b = to_grid(x1, y1), to_grid(x2, y2)
        if a not in coord_set:
            warnings.append(f"连线端点 {a} 无对应节点")
        if b not in coord_set:
            warnings.append(f"连线端点 {b} 无对应节点")
        edges.append([list(a), list(b)])

    return {
        "template_id": f"Layer_{LAYER}-{idx}",
        "title": title,
        "nodes": nodes,
        "edges": edges,
    }, warnings


def main():
    global MAX_Y
    with open(SRC, "r", encoding="utf-8") as f:
        html = f.read()

    # 自动推导 MAX_Y：取全图最大 cy（换层图无需手改）
    all_cy = [float(m) for m in re.findall(r'cy="([-\d.]+)"', html)]
    if all_cy:
        MAX_Y = int(round(max(all_cy)))
        print(f">>> 自动检测 MAX_Y = {MAX_Y} (y 范围 0~{MAX_Y})")

    blocks = BUTTON_RE.findall(html)
    new_cards = []
    for i, block in enumerate(blocks, 1):
        card, warns = build_card(block, i)
        new_cards.append(card)
        status = f"[{card['template_id']}] {card['title']} | 节点 {len(card['nodes'])} 连线 {len(card['edges'])}"
        print(status)
        for w in warns:
            print(f"    [!] {w}")

    # 追加写入：读入已有模板，按 template_id 合并(同 id 更新，其它层保留)
    existing = []
    if os.path.exists(OUT):
        try:
            with open(OUT, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            print(f"[警告] 读取已有 {OUT} 失败，将重新创建: {e}")
            existing = []

    new_ids = {c["template_id"] for c in new_cards}
    merged = [c for c in existing if c.get("template_id") not in new_ids] + new_cards

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n本次写入 {len(new_cards)} 个模板，文件现共 {len(merged)} 个 -> {OUT}")


if __name__ == "__main__":
    main()
