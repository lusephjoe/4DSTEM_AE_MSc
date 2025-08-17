"""
SVG Renderer for Publication-Grade Autoencoder Diagrams

Creates professional 3D-style diagrams with vertical cuboids, rotated text,
custom icons, and precise layout control.
"""

import math
import random
from typing import Dict, List, Tuple, Optional
import svgwrite


PX = float


class SVGRenderer:
    """SVG renderer for publication-grade autoencoder diagrams."""
    
    def __init__(self, width: int = 1800, height: int = 1200, font: str = "Inter, Arial, sans-serif"):
        self.dw = svgwrite.Drawing(size=(width, height), profile="tiny")
        self.font = font
        self.width, self.height = width, height
        self._gradient_cache = set()

    # ---------- Low-level drawing primitives ----------
    
    def arrow(self, p1: Tuple[PX, PX], p2: Tuple[PX, PX], 
              color: str = "#6B7280", stroke_width: PX = 2.4, head: PX = 10, shadow: bool = False):
        """Draw arrow from p1 to p2 with triangular head."""
        if shadow:
            self.dw.add(self.dw.line(p1, p2, stroke="#000", stroke_width=stroke_width+2, opacity=0.08))
        line = self.dw.add(self.dw.line(p1, p2, stroke=color, stroke_width=stroke_width))
        
        # Calculate arrow head
        ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        tip = p2
        left = (p2[0] - head * math.cos(ang) + head * 0.6 * math.sin(ang),
                p2[1] - head * math.sin(ang) - head * 0.6 * math.cos(ang))
        right = (p2[0] - head * math.cos(ang) - head * 0.6 * math.sin(ang),
                 p2[1] - head * math.sin(ang) + head * 0.6 * math.cos(ang))
        self.dw.add(self.dw.polygon([tip, left, right], fill=color))

    def rounded_rect(self, x: PX, y: PX, w: PX, h: PX, r: PX = 10, 
                     fill: str = "#fff", stroke: str = "#111", sw: PX = 1.5):
        """Draw rounded rectangle."""
        return self.dw.add(self.dw.rect(insert=(x, y), size=(w, h), rx=r, ry=r,
                                        fill=fill, stroke=stroke, stroke_width=sw))

    def label(self, x: PX, y: PX, text: str, size: int = 14, 
              color: str = "#111", anchor: str = "middle", weight: str = "normal"):
        """Draw text label."""
        return self.dw.add(self.dw.text(text, insert=(x, y),
                                        fill=color, font_family=self.font,
                                        font_size=size, text_anchor=anchor,
                                        font_weight=weight))

    # ---------- 3D Glyph drawing ----------
    
    def cuboid(self, x: PX, y: PX, w: PX, h: PX, color: str = "#3B82F6",
               title: str = "Block", meta: str = "", rotate_text: bool = True,
               title_size: int = 24, meta_size: int = 18,
               caption: Optional[str] = None, caption_size: int = 14) -> Dict[str, Tuple[PX, PX]]:
        g = self.dw.g()

        depth = 26              # ← more "sheet-like" depth
        overlap = 6             # ← slight overlap kills tiny seams

        # front face
        front = self.dw.rect((x, y), (w, h), rx=14, ry=14,
                             fill=f"url(#{self._gradient(color)})",
                             stroke="#111", stroke_width=1.4,
                             stroke_linejoin="round")

        # top face (extends slightly beyond front to avoid corner gap)
        top = self.dw.polygon([
            (x + 10 - overlap, y),
            (x + w - 10 + overlap, y),
            (x + w - 10 + depth + overlap, y - depth),
            (x + 10 + depth - overlap, y - depth)
        ], fill=self._tint(color, 1.2), stroke="#111", stroke_width=1.1)

        # side face
        side = self.dw.polygon([
            (x + w,           y + 10 - overlap),
            (x + w,           y + h - 10 + overlap),
            (x + w + depth,   y + h - 10 - depth + overlap),
            (x + w + depth,   y + 10 - depth - overlap)
        ], fill=self._tint(color, 0.9), stroke="#111", stroke_width=1.1)

        # draw order
        g.add(side); g.add(top); g.add(front)

        # continuous outline (ensures no gaps anywhere)
        outline = self.dw.path(
            d=("M {x1},{y1} L {x2},{y2} L {x3},{y3} L {x4},{y4} "
               "L {x5},{y5} L {x6},{y6} L {x7},{y7}").format(
                x1=x+10-overlap, y1=y,
                x2=x+w-10+overlap, y2=y,
                x3=x+w-10+depth+overlap, y3=y-depth,
                x4=x+w+depth, y4=y+10-depth-overlap,
                x5=x+w, y5=y+10-overlap,
                x6=x+w, y6=y+h-10+overlap,
                x7=x+w+depth, y7=y+h-10-depth+overlap),
            fill="none", stroke="#111", stroke_width=1.0)
        g.add(outline)

        # rotated labels, centered on FRONT face
        tx, ty = x + w/2, y + h/2
        if rotate_text:
            tsize = max(16, title_size - 2 if len(title) > 18 else title_size)
            tg = self.dw.g(transform=f"rotate(-90,{tx},{ty})")
            tg.add(self.dw.text(title, insert=(tx, ty-18), font_family=self.font,
                                font_size=tsize, font_weight="bold",
                                text_anchor="middle", fill="#0B1021"))
            if meta:
                tg.add(self.dw.text(meta, insert=(tx, ty+20), font_family=self.font,
                                    font_size=meta_size, text_anchor="middle",
                                    fill="#374151"))
            g.add(tg)
        else:
            self.label(tx, ty-6, title, size=title_size, weight="bold")
            if meta:
                self.label(tx, ty+18, meta, size=meta_size, color="#374151")

        self.dw.add(g)

        # caption under block (channels)
        if caption:
            self.label(x + w/2, y + h + 22, caption, size=caption_size, color="#374151")

        return {"left": (x - 6, y + h/2), "right": (x + w + depth, y + h/2)}

    def latent_glyph(self, cx: PX, cy: PX, rx: PX = 48, ry: PX = 38, 
                     color: str = "#8B5CF6", z_text: str = "z = 32"):
        g = self.dw.g()

        # radial gradient
        gid = f"radgrad_{color.strip('#')}"
        if gid not in self._gradient_cache:
            rg = self.dw.radialGradient(id=gid, center=("50%", "50%"))
            rg.add_stop_color(0, self._tint(color, 1.6), opacity=0.9)
            rg.add_stop_color(1, "#FFFFFF", opacity=1.0)
            self.dw.defs.add(rg); self._gradient_cache.add(gid)

        g.add(self.dw.ellipse(center=(cx, cy), r=(rx, ry),
                              fill=f"url(#{gid})", stroke="#111", stroke_width=1.2))

        # subtle dashed contours
        for k, op in [(0.75, .25), (0.55, .18)]:
            g.add(self.dw.ellipse(center=(cx, cy), r=(rx*k, ry*k),
                                  fill="none", stroke="#6D28D9",
                                  stroke_dasharray="4,6", opacity=op))

        # deterministic point cloud
        rnd = random.Random(42)
        for _ in range(90):
            px = cx + rnd.uniform(-rx*0.8, rx*0.8)
            py = cy + rnd.uniform(-ry*0.6, ry*0.6)
            r  = 1.2 if rnd.random() > 0.88 else 0.9
            g.add(self.dw.circle((px,py), r=r, fill="#111", opacity=0.45))

        self.label(cx, cy + ry + 18, z_text, size=14, color="#374151")
        self.dw.add(g)
        return {"left": (cx - rx - 6, cy), "right": (cx + rx + 6, cy)}

    def icon_4dstem(self, x: PX, y: PX, w: PX = 110, h: PX = 160, title: str = "4D-STEM"):
        g = self.dw.g()
        # stack of scan frames, gently offset
        stack_w, stack_h = w * 0.46, h * 0.58
        for i, off in enumerate([16, 12, 8, 4, 0]):
            opacity = 0.28 + i*0.12
            g.add(self.dw.rect((x+off, y+off), (stack_w, stack_h),
                               fill="#FFFFFF", stroke="#CBD5E1", rx=8, ry=8, opacity=opacity))
        # subtle grid on top frame
        cell = 10
        grid = self.dw.g(stroke="#E5E7EB", stroke_width=0.6)
        for gx in range(int(stack_w//cell)):
            grid.add(self.dw.line((x+0+gx*cell, y+0), (x+0+gx*cell, y+stack_h)))
        for gy in range(int(stack_h//cell)):
            grid.add(self.dw.line((x+0, y+0+gy*cell), (x+stack_w, y+0+gy*cell)))
        g.add(grid)

        # detector panel
        det_x, det_w, det_h = x + w*0.56, w*0.40, h*0.92
        det = self.dw.rect((det_x, y+6), (det_w, det_h), fill="#FFFFFF",
                           stroke="#CBD5E1", rx=10, ry=10)
        g.add(det)
        # diffraction rings/spots
        cx, cy = det_x + det_w/2, y + 6 + det_h/2
        for r, op in [(det_w*0.15, 0.22), (det_w*0.28, 0.18), (det_w*0.38, 0.12)]:
            g.add(self.dw.circle((cx, cy), r=r, fill="none", stroke="#9CA3AF", opacity=op, stroke_width=1))
        for px, py in [(cx-18, cy-12), (cx+10, cy-22), (cx-6, cy+14), (cx+16, cy+6)]:
            g.add(self.dw.circle((px, py), r=2.6, fill="#111", opacity=0.55))

        # tiny arrow indicating scanning → detector
        self.arrow((x+stack_w+8, y+stack_h/2), (det_x-6, y+stack_h/2), color="#9CA3AF", stroke_width=1.8, head=7)

        self.label(x + w/2, y + h + 16, title, size=13, color="#374151")
        self.dw.add(g)
        return {"left": (x, y + h/2), "right": (x + w, y + h/2)}

    # ---------- Panel and layout components ----------
    
    def panel_frame(self, x: PX, y: PX, w: PX, h: PX, title: str):
        """Draw dashed panel frame with title."""
        self.dw.add(self.dw.rect((x, y), (w, h), rx=12, ry=12,
                                 fill="none", stroke="#93C5FD", 
                                 stroke_dasharray="6,6", stroke_width=1.5))
        self.label(x + 10, y - 6, title, size=14, color="#1F2937", 
                   anchor="start", weight="bold")

    def ghost_block(self, x: PX, y: PX, w: PX = 80, h: PX = 40, label_text: str = "Previous"):
        """Draw ghost block for micro-diagrams."""
        self.rounded_rect(x, y, w, h, r=10, fill="#fff", stroke="#D1D5DB", sw=1.2)
        self.label(x + w / 2, y + h / 2 + 4, label_text, size=12, color="#6B7280")
        return {"left": (x, y + h / 2), "right": (x + w, y + h / 2)}

    def plus_junction(self, x: PX, y: PX, r: PX = 10, color: str = "#10B981"):
        """Draw '+' junction for skip connections."""
        self.rounded_rect(x - r, y - r, 2 * r, 2 * r, r=r, fill=color, stroke=color, sw=1.0)
        self.label(x, y + 4, "+", size=12, color="#fff", weight="bold")

    # ---------- Utility methods ----------
    
    def _gradient(self, base_color: str) -> str:
        """Create or get cached gradient for color."""
        gid = f"grad_{base_color.strip('#')}"
        if gid in self._gradient_cache:
            return gid
        
        lg = self.dw.linearGradient(id=gid, start=("0%", "0%"), end=("0%", "100%"))
        lg.add_stop_color(0, self._tint(base_color, 1.25))
        lg.add_stop_color(1, "#FFFFFF")
        self.dw.defs.add(lg)
        self._gradient_cache.add(gid)
        return gid

    def _tint(self, hexcolor: str, factor: float = 1.1) -> str:
        """Lighten or darken hex color by factor."""
        c = hexcolor.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}".upper()

    def save(self, filename: str):
        """Save SVG to file."""
        self.dw.saveas(filename)


def compose_publication_svg(blocks: List, out_path: str = "publication.svg",
                            input_icon: bool = True, output_icon: bool = True, 
                            z_label: str = "z") -> str:
    """Compose complete publication figure with three panels."""
    R = SVGRenderer(width=1800, height=1200)
    R.dw.add(R.dw.rect((0, 0), (R.width, R.height), fill="white"))   # white bg

    # === Panel (a): Model Overview ===
    panel_x, panel_y = 40, 40
    panel_w, panel_h = 1720, 360
    R.panel_frame(panel_x, panel_y, panel_w, panel_h, "a) Model Structure")

    # geometry (denser layout)
    x0 = panel_x + 200
    y = panel_y + 36
    W, H = 68, 280             # wider, taller blocks
    gap = 40                   # tight spacing
    cursor_x = x0

    # 4D-STEM icons (aligned horizontally with blocks)
    input_anchor = None
    output_anchor = None
    icon_y = y + (H - 160) // 2  # center icons vertically with blocks
    if input_icon:
        input_anchor = R.icon_4dstem(panel_x + 12, icon_y, w=120, h=160, title="4D-STEM Input")
    if output_icon:
        output_anchor = R.icon_4dstem(panel_x + panel_w - 210, icon_y, w=120, h=160, title="Reconstruction")

    anchors = []
    last_right = None

    def place_right_of(prev_right_x, span):
        """return the x for the next block's left so gap is uniform"""
        return prev_right_x + gap

    for i, b in enumerate(blocks):
        # color & labels
        color = {"encoder":"#3B82F6","decoder":"#F59E0B","misc":"#D1D5DB"}.get(b.group, "#D1D5DB")
        if len(b.out_shape) == 4:
            meta = f"{b.out_shape[2]}×{b.out_shape[3]}"
            caption = f"{b.in_shape[1]}→{b.out_shape[1]} ch"
        elif len(b.out_shape) == 2:
            meta = f"dim={b.out_shape[1]}"; caption = None
        else:
            meta = ""; caption = None

        if b.type == "bottleneck":
            rx, ry = 52, 42  # larger latent glyph
            # place latent with proper spacing
            if last_right is None:
                cx = x0 + rx
            else:
                latent_left = last_right[0] + gap
                cx = latent_left + rx + 6
            anchor = R.latent_glyph(cx, y + H/2, rx=rx, ry=ry, z_text=b.name)
            anchors.append(("latent", anchor))
            last_right = anchor["right"]
            continue

        # non-latent block: compute x from previous right anchor
        x = x0 if last_right is None else place_right_of(last_right[0], W)
        anchor = R.cuboid(x, y, W, H, color=color, title=b.name,
                          meta=meta, rotate_text=True,
                          title_size=24, meta_size=18,
                          caption=caption, caption_size=14)
        anchors.append((b.id, anchor))
        last_right = anchor["right"]

    # Draw arrows connecting blocks
    if input_anchor and anchors:
        # From input icon to first block
        start = input_anchor["right"]
        first_left = anchors[0][1]["left"]
        R.arrow(start, first_left, color="#6B7280", shadow=True)

    # Between blocks
    for i in range(len(anchors) - 1):
        right_i = anchors[i][1]["right"]
        left_j = anchors[i + 1][1]["left"]
        R.arrow(right_i, left_j, color="#6B7280", shadow=True)

    # From last block to output icon
    if output_anchor and anchors:
        last_right = anchors[-1][1]["right"]
        end = output_anchor["left"]
        R.arrow(last_right, end, color="#6B7280", shadow=True)

    # === Panel (b): ResNet MaxPool Block ===
    bx, by = 40, panel_y + panel_h + 40
    bw, bh = 1720, 300
    R.panel_frame(bx, by, bw, bh, "b) ResNet MaxPool Block")

    # Micro-diagram specification
    down_spec = [
        ("Conv", "128ch"), ("Conv", "128ch"), ("Norm", ""),
        ("ReLU", ""), ("Conv", "128ch"), ("Norm", ""), ("ReLU", ""), ("Pool", "")
    ]
    
    x = bx + 110
    y = by + 32
    W, H = 64, 220  # bigger micro-diagram blocks
    gap = 38
    
    # Previous layer ghost
    prev_ghost = R.ghost_block(bx + 10, y + 60, label_text="Previous")
    prev_right = prev_ghost["right"]

    plus_pos = None
    block_anchors = []
    
    for i, (op, meta) in enumerate(down_spec):
        # Color mapping for operations
        op_colors = {
            "Conv": "#F6D6A8", "Norm": "#A7E1F5", "ReLU": "#D9C7C0", 
            "Pool": "#B8E2B1", "Up": "#CFE399"
        }
        color = op_colors.get(op, "#E5E7EB")
        
        # Draw operation cuboid
        cap = meta if op in ("Conv", "Up", "Pool") and meta else None
        anchor = R.cuboid(x, y, W, H, color=color, title=op, meta="",
                          rotate_text=True, title_size=22, meta_size=16,
                          caption=cap, caption_size=14)
        block_anchors.append(anchor)
        
        # Draw main path arrow
        R.arrow(prev_right, anchor["left"])
        prev_right = anchor["right"]
        
        # Mark skip connection point (at Conv layer 5)
        if op == "Conv" and i == 4:
            plus_pos = (anchor["right"][0] + 20, y + H//2)
        
        x += W + gap

    # Next layer ghost
    next_ghost = R.ghost_block(x, y + 60, label_text="Next")
    R.arrow(prev_right, next_ghost["left"])

    # Draw skip connection
    if plus_pos:
        # Skip arc from previous layer to plus junction
        skip_start = (bx + 10 + 80, y + H//2)
        R.arrow(skip_start, (plus_pos[0] - 16, plus_pos[1]), color="#10B981")
        
        # Plus junction
        R.plus_junction(plus_pos[0], plus_pos[1])
        
        # From plus to next layer
        R.arrow((plus_pos[0] + 10, plus_pos[1]), next_ghost["left"], color="#10B981")

    # === Panel (c): ResNet UpSample Block ===
    cx, cy = 40, by + bh + 40
    cw, ch = 1720, 300
    R.panel_frame(cx, cy, cw, ch, "c) ResNet UpSample Block")

    # UpSample specification (Up first for logical flow)
    up_spec = [
        ("Up", "×2"), ("Conv", "128ch"), ("Conv", "128ch"), ("Norm", ""),
        ("ReLU", ""), ("Conv", "128ch"), ("Norm", ""), ("ReLU", "")
    ]
    
    x = cx + 110
    y = cy + 32
    
    # Previous layer ghost
    prev_ghost = R.ghost_block(cx + 10, y + 60, label_text="Previous")
    prev_right = prev_ghost["right"]

    plus_pos = None
    
    for i, (op, meta) in enumerate(up_spec):
        color = op_colors.get(op, "#E5E7EB")
        title = "Up" if op == "Up" else op
        
        # Draw operation cuboid
        cap = meta if op in ("Conv", "Up", "Pool") and meta else None
        anchor = R.cuboid(x, y, W, H, color=color, title=title, meta="",
                          rotate_text=True, title_size=22, meta_size=16,
                          caption=cap, caption_size=14)
        
        # Draw main path arrow
        R.arrow(prev_right, anchor["left"])
        prev_right = anchor["right"]
        
        # Mark skip connection point (at Conv layer 2, after Up)
        if op == "Conv" and i == 5:  # adjusted for Up-first ordering
            plus_pos = (anchor["right"][0] + 20, y + H//2)
        
        x += W + gap

    # Next layer ghost
    next_ghost = R.ghost_block(x, y + 60, label_text="Next")
    R.arrow(prev_right, next_ghost["left"])

    # Draw skip connection for UpSample
    if plus_pos:
        skip_start = (cx + 10 + 80, y + H//2)
        R.arrow(skip_start, (plus_pos[0] - 16, plus_pos[1]), color="#10B981")
        R.plus_junction(plus_pos[0], plus_pos[1])
        R.arrow((plus_pos[0] + 10, plus_pos[1]), next_ghost["left"], color="#10B981")

    # Save the final SVG
    R.save(out_path)
    return out_path