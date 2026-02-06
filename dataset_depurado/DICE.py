import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from shapely.ops import unary_union


def parse_alto_textblocks(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"alto": root.tag.split("}")[0].strip("{")}

    polys = []

    for block in root.findall(".//alto:TextBlock", ns):

        polygon = block.find(".//alto:Shape/alto:Polygon", ns)
        if polygon is None:
            continue

        pts_raw = polygon.attrib.get("POINTS", "").strip().split()
        if not pts_raw:
            continue

        pts = []
        for i in range(0, len(pts_raw), 2):
            x = float(pts_raw[i])
            y = float(pts_raw[i+1])
            pts.append((x, y))

        polys.append(Polygon(pts))

    return polys

def parse_pagexml_textlines(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"p": root.tag.split("}")[0].strip("{")}

    polys = []

    for line in root.findall(".//p:TextLine", ns):
        coords = line.find("p:Coords", ns)
        if coords is None:
            continue

        pts_raw = coords.attrib["points"].split()
        pts = []

        for pnt in pts_raw:
            x, y = pnt.split(",")
            pts.append((float(x), float(y)))

        polys.append(Polygon(pts))

    return polys


def parse_pagexml_textblocks(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"p": root.tag.split("}")[0].strip("{")}

    textline_polys = []

    for line in root.findall(".//p:TextLine", ns):
        coords = line.find("p:Coords", ns)
        if coords is None:
            continue

        pts = []
        for pnt in coords.attrib["points"].split():
            x, y = pnt.split(",")
            pts.append((float(x), float(y)))

        poly = Polygon(pts)

        
        if not poly.is_valid:
            poly = poly.buffer(0)

        if not poly.is_empty:
            textline_polys.append(poly)

    if not textline_polys:
        return []

    merged = unary_union(textline_polys)

    if isinstance(merged, Polygon):
        return [merged]

    elif isinstance(merged, MultiPolygon):
        return list(merged.geoms)

    else:
        return []


# ============================================================
# 1. INTERSECCIONES OCR–GT IoU
# ============================================================

def compute_intersections(ocr_polys, gt_polys, iou_threshold=0.5):
    matches = []

    for o in ocr_polys:
        for g in gt_polys:
            inter = o.intersection(g)

            if inter.is_empty:
                continue

            inter_area = inter.area
            union_area = o.union(g).area
            iou = inter_area / union_area if union_area > 0 else 0

            matches.append((o, g, inter, iou))

    return matches

# ============================================================
# 2. INTERSECCIONES OCR–GT DICE
# ============================================================

def DICE_Index(textblock_A, textblock_B):
    if not isinstance(textblock_A, (list, tuple)):
        textblock_A = [textblock_A]
    if not isinstance(textblock_B, (list, tuple)):
        textblock_B = [textblock_B]

    union_a = unary_union(textblock_A).buffer(0)
    union_b = unary_union(textblock_B).buffer(0)

    intersection = union_a.intersection(union_b)

    area_a = union_a.area
    area_b = union_b.area
    area_intersection = intersection.area

    dice = 0.0 if (area_a + area_b) == 0 else (2 * area_intersection) / (area_a + area_b)


    return area_a,area_b, area_intersection, dice

def plot_polygons_ocr(ocr_polys):
    fig, ax = plt.subplots(figsize=(10, 14))

    for poly in ocr_polys:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3)

    ax.set_title("Polígonos OCR")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.show()


def plot_polygons_gt(gt_polys):
    fig, ax = plt.subplots(figsize=(10, 14))

    for poly in gt_polys:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3)

    ax.set_title("Polígonos GT")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.show()

def plot_intersections(matches):

    fig, ax = plt.subplots(figsize=(10, 14))

    for (o, g, inter, iou) in matches:
        if inter.is_empty:
            continue

        if inter.geom_type == "Polygon":
            geoms = [inter]
        else:  # MultiPolygon
            geoms = list(inter.geoms)

        color = "green" if iou >= 0.5 else "red"

        for geom in geoms:
            if geom.is_empty or geom.area == 0:
                continue

            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.45, color=color)

    ax.set_title("Intersecciones OCR vs GT (verde = bueno, rojo = malo)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.show()

def visualize_textblocks(textblocks, figsize=(10, 14)):
    fig, ax = plt.subplots(figsize=figsize)

    for poly in textblocks:
        if poly.is_empty:
            continue

        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.45, linewidth=2)
        ax.plot(x, y, linewidth=2)

    ax.set_aspect("equal")
    ax.invert_yaxis()  
    ax.set_title("TextBlocks")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)

    plt.show()

# ============================================================
# 1. Visualización OCR-GT Textblock v Textline 
# ============================================================

def plot_venn_overlay(ocr_polys, gt_polys, matches):
    fig, ax = plt.subplots(figsize=(10, 14))

    for poly in ocr_polys:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.25, color="#1f77b4", edgecolor="black", linewidth=1)

    for poly in gt_polys:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.25, color="#ff7f0e", edgecolor="black", linewidth=1)

    for (o, g, inter, iou) in matches:
        if inter.is_empty:
            continue

        geoms = [inter] if inter.geom_type == "Polygon" else list(inter.geoms)
        color = "green" if iou >= 0.5 else "red"

        for geom in geoms:
            if geom.is_empty:
                continue
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.45, color=color, edgecolor="black", linewidth=1)

    ax.set_title("OCR vs GT — Diagrama tipo Venn (Azul=OCR, Naranja=GT, Verde/Rojo=Intersecciones)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.show()

# ============================================================
# 2. Visualización OCR-GT Textblock v Textblock 
# ============================================================

def visualize_textblock_overlap(textblocks_a, textblocks_b, figsize=(10, 14)):
    fig, ax = plt.subplots(figsize=figsize)

    union_a = unary_union(textblocks_a).buffer(0)
    union_b = unary_union(textblocks_b).buffer(0)

    intersection = union_a.intersection(union_b)

    area_a = union_a.area
    area_b = union_b.area
    area_intersection = intersection.area

    dice = 0.0 if (area_a + area_b) == 0 else (2 * area_intersection) / (area_a + area_b)
    intersection_pct = 0.0 if area_b == 0 else (area_intersection / area_b) * 100

   
    if not union_a.is_empty:
        geoms = [union_a] if union_a.geom_type == "Polygon" else union_a.geoms
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="royalblue", alpha=0.4)
            ax.plot(x, y, color="royalblue", linewidth=2)

    
    if not union_b.is_empty:
        geoms = [union_b] if union_b.geom_type == "Polygon" else union_b.geoms
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="red", alpha=0.4)
            ax.plot(x, y, color="red", linewidth=2)

    
    if not intersection.is_empty:
        geoms = [intersection] if intersection.geom_type == "Polygon" else intersection.geoms
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="lightgreen", alpha=0.7)
            ax.plot(x, y, color="lightgreen", linewidth=2)

   
    ax.text(
        0.02, 0.98,
        f"DICE: {dice:.3f}\nIntersección: {intersection_pct:.1f}%",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Comparación de TextBlocks (A vs B)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)

    plt.show()

    return dice, intersection_pct

def main(OCR_ALTO, GT_PAGE):
    print("Cargando OCR (ALTO)...")
    ocr_polys = parse_alto_textblocks(OCR_ALTO)

    print("Cargando Ground Truth (PAGE XML)...")
    gt_polys = parse_pagexml_textlines(GT_PAGE)

    text_block_gt = parse_pagexml_textblocks(GT_PAGE)

    print("Calculando coincidencias text line vs text block")
    matches_textblock_vs_textline = compute_intersections(ocr_polys, gt_polys)

    print("Calculando coincidencias text block OCR vs text block GT")
    matches_textblock_vs_textblock = compute_intersections(ocr_polys, text_block_gt)


    print(f"Total de coincidencias encontradas textline v textblock: {len(matches_textblock_vs_textline)}")
    print(f"Total de coincidencias encontradas textblock v textblock: {len(matches_textblock_vs_textblock)}")

    print(DICE_Index(ocr_polys, text_block_gt))
    plot_polygons_ocr(ocr_polys)
    plot_polygons_gt(gt_polys)
    plot_intersections(matches_textblock_vs_textline)
    plot_venn_overlay(ocr_polys, gt_polys, matches_textblock_vs_textline)
    visualize_textblocks(text_block_gt)
    visualize_textblock_overlap(ocr_polys, text_block_gt)



if __name__ == "__main__":
    OCR_ALTO = "./Results/Segmentation/segments/segmentation_output_0001_AGN_Tierras_vol38_exp2_f22o136.xml"
    GT_PAGE = "./processed_data/Segmentation/test/0001_AGN_Tierras_vol38_exp2_f22o136.xml"
    main(OCR_ALTO, GT_PAGE) 