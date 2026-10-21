"""
Generate comprehensive, publication-grade Draw.io XML diagram for BiRTrack.
Covers:
1. Overall Architecture & Tensor Shapes
2. Rep-BLGR Detailed Circuit & Algebraic Reparameterization Math
3. 2-Phase Training Pipeline (SA-ORR & MI-AFKD)
4. TensorRT Edge Deployment & Graph Optimization
"""
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os

def create_drawio_diagram():
    mxfile = ET.Element('mxfile', {
        'host': 'app.diagrams.net',
        'modified': '2026-10-21T12:00:00.000Z',
        'agent': 'Antigravity AI Agent',
        'version': '21.0.0',
        'type': 'device'
    })

    # Helper function to add nodes and edges
    class DiagramBuilder:
        def __init__(self, page_id, page_name, width=2300, height=1300):
            self.diagram = ET.SubElement(mxfile, 'diagram', {'id': page_id, 'name': page_name})
            self.model = ET.SubElement(self.diagram, 'mxGraphModel', {
                'dx': '1422', 'dy': '794', 'grid': '1', 'gridSize': '10',
                'guides': '1', 'tooltips': '1', 'connect': '1', 'arrows': '1',
                'fold': '1', 'page': '1', 'pageScale': '1',
                'pageWidth': str(width), 'pageHeight': str(height),
                'math': '1', 'shadow': '0'
            })
            self.root = ET.SubElement(self.model, 'root')
            ET.SubElement(self.root, 'mxCell', {'id': '0'})
            ET.SubElement(self.root, 'mxCell', {'id': '1', 'parent': '0'})
            self.cell_count = 2

        def next_id(self):
            cid = f"cell_{self.cell_count}"
            self.cell_count += 1
            return cid

        def add_box(self, x, y, w, h, value, style, parent='1', cell_id=None):
            cid = cell_id or self.next_id()
            cell = ET.SubElement(self.root, 'mxCell', {
                'id': cid,
                'value': value,
                'style': style,
                'vertex': '1',
                'parent': parent
            })
            ET.SubElement(cell, 'mxGeometry', {
                'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'as': 'geometry'
            })
            return cid

        def add_edge(self, source, target, value='', style=None, points=None):
            cid = self.next_id()
            default_style = (
                'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;'
                'strokeColor=#334155;strokeWidth=2;fontSize=11;fontColor=#1E293B;'
            )
            st = style if style is not None else default_style
            cell = ET.SubElement(self.root, 'mxCell', {
                'id': cid,
                'value': value,
                'style': st,
                'edge': '1',
                'parent': '1',
                'source': source,
                'target': target
            })
            geom = ET.SubElement(cell, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
            if points:
                pts_elem = ET.SubElement(geom, 'Array', {'as': 'points'})
                for px, py in points:
                    ET.SubElement(pts_elem, 'mxPoint', {'x': str(px), 'y': str(py)})
            return cid

    # =========================================================================
    # PAGE 1: Overall BiRTrack Architecture
    # =========================================================================
    p1 = DiagramBuilder('p1_overall', '1. BiRTrack Overall Architecture', width=2360, height=1320)

    # Header Banner
    p1.add_box(
        40, 30, 2280, 95,
        '<div style="font-size: 22px; font-weight: bold; color: #0F172A; margin-bottom: 4px;">'
        'BiRTrack: Reparameterized Bidirectional Local-Global Refiner &amp; Search-Aware Adaptive Distillation'
        '</div>'
        '<div style="font-size: 13px; color: #334155;">'
        'End-to-End UAV Tracking Architecture: Single-Stream ViT (l=8) &bull; Structural Reparameterization (Rep-BLGR) &bull; Zero-Inference Training (SA-ORR &amp; MI-AFKD) &bull; 285+ FPS on RTX / 75+ FPS on Orin Nano'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;shadow=1;'
    )

    # Column 1: Input & Patch Embeddings
    col1 = p1.add_box(
        40, 145, 400, 1130,
        '<b>1. INPUTS &amp; PATCH EMBEDDING</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=14;fontColor=#1E3A8A;rounded=1;shadow=1;'
    )
    z_img = p1.add_box(
        70, 200, 340, 100,
        '<b>Template Image Z</b><br>'
        '<span style="font-size:11px; color:#475569;">B &times; 3 &times; 128 &times; 128</span><br>'
        '<span style="font-size:10px; color:#2563EB;">Target Crop (Frame 0 or Reference)</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=2;fontSize=12;fontColor=#1E40AF;',
        parent=col1
    )
    x_img = p1.add_box(
        70, 330, 340, 100,
        '<b>Search Image X</b><br>'
        '<span style="font-size:11px; color:#475569;">B &times; 3 &times; 256 &times; 256</span><br>'
        '<span style="font-size:10px; color:#059669;">Current Frame Search Region</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#D1FAE5;strokeColor=#059669;strokeWidth=2;fontSize=12;fontColor=#065F46;',
        parent=col1
    )
    pe_z = p1.add_box(
        70, 470, 340, 80,
        '<b>Patch Embedding (Z)</b><br>'
        '<span style="font-size:11px; color:#475569;">Conv2d(3, 192, k=16, s=16)</span><br>'
        '<span style="font-size:10px; color:#1E40AF;">Tokens: K<sub>z</sub> = 64 &times; 192</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#3B82F6;strokeWidth=1.5;fontSize=12;',
        parent=col1
    )
    pe_x = p1.add_box(
        70, 580, 340, 80,
        '<b>Patch Embedding (X)</b><br>'
        '<span style="font-size:11px; color:#475569;">Conv2d(3, 192, k=16, s=16)</span><br>'
        '<span style="font-size:10px; color:#059669;">Tokens: K<sub>x</sub> = 256 &times; 192</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1.5;fontSize=12;',
        parent=col1
    )
    pos_emb = p1.add_box(
        70, 690, 340, 85,
        '<b>Add Learnable Position Embeddings</b><br>'
        '<span style="font-size:10px; color:#475569;">Pos_Emb_Z: [1, 64, 192]<br>Pos_Emb_X: [1, 256, 192]<br>+ Pos_Dropout(p=0.0)</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;fontSize=11;fontColor=#92400E;',
        parent=col1
    )
    concat_node = p1.add_box(
        70, 810, 340, 75,
        '<b>Concat Tokens [Z, X]</b><br>'
        '<span style="font-size:12px; font-weight:bold; color:#1E293B;">Tensor: [B, 320, 192]</span><br>'
        '<span style="font-size:10px; color:#64748B;">Unified Single-Stream Sequence</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#475569;strokeWidth=2;fontSize=11;',
        parent=col1
    )
    design_note1 = p1.add_box(
        70, 920, 340, 150,
        '<b>Key Engineering Property:</b><br>'
        '<div style="text-align:left; font-size:10px; line-height:1.4; color:#334155; margin-top:4px;">'
        '&bull; Fixed patch size: 16 &times; 16<br>'
        '&bull; No dynamic sequence lengths<br>'
        '&bull; Constant memory stride<br>'
        '&bull; Zero unstructured RAM access<br>'
        '&bull; 100% deterministic static tensor shapes'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#94A3B8;strokeWidth=1;fontSize=11;',
        parent=col1
    )

    p1.add_edge(z_img, pe_z)
    p1.add_edge(x_img, pe_x)
    p1.add_edge(pe_z, pos_emb)
    p1.add_edge(pe_x, pos_emb)
    p1.add_edge(pos_emb, concat_node)

    # Column 2: Single-Stream Static ViT Backbone
    col2 = p1.add_box(
        470, 145, 450, 1130,
        '<b>2. STATIC ViT BACKBONE (l = 8 BLOCKS)</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#F5F3FF;strokeColor=#7C3AED;strokeWidth=2;fontSize=14;fontColor=#5B21B6;rounded=1;shadow=1;'
    )
    vit_in = p1.add_box(
        500, 200, 390, 50,
        '<b>Input Tokens: [B, 320, 192]</b>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#EDE9FE;strokeColor=#8B5CF6;strokeWidth=1.5;fontSize=11;fontColor=#4C1D95;',
        parent=col2
    )
    blk1 = p1.add_box(
        500, 270, 390, 90,
        '<b>Transformer Block 1</b><br>'
        '<span style="font-size:10px; color:#475569;">Pre-LN &rarr; Multi-Head Self-Attention (MHSA, h=3) &rarr; + Residual<br>'
        'Pre-LN &rarr; MLP (192 &rarr; 768 &rarr; 192, GELU) &rarr; + Residual</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#A78BFA;strokeWidth=1.5;fontSize=11;',
        parent=col2
    )
    blk2 = p1.add_box(
        500, 380, 390, 90,
        '<b>Transformer Block 2</b><br>'
        '<span style="font-size:10px; color:#475569;">Joint Attention between Template &amp; Search Tokens<br>'
        'Cross-correlation &amp; feature extraction unified in 1 stream</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#A78BFA;strokeWidth=1.5;fontSize=11;',
        parent=col2
    )
    blk_dots = p1.add_box(
        500, 490, 390, 50,
        '<b>&bull; &bull; &bull; (Blocks 3 to 7: Static Depth) &bull; &bull; &bull;</b>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#C4B5FD;strokeWidth=1;dashed=1;fontSize=11;fontColor=#6B21A8;',
        parent=col2
    )
    blk8 = p1.add_box(
        500, 560, 390, 90,
        '<b>Transformer Block 8 (Active Depth l=8)</b><br>'
        '<span style="font-size:10px; color:#475569;">Structured Trimming from Teacher (L=12 &rarr; l=8)<br>'
        'Zero control flow (no if/else, no dynamic scatter/gather)</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#7C3AED;strokeWidth=2;fontSize=11;fontColor=#4C1D95;',
        parent=col2
    )
    ln_final = p1.add_box(
        500, 670, 390, 50,
        '<b>Final LayerNorm(192)</b> &rarr; [B, 320, 192]',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#EDE9FE;strokeColor=#8B5CF6;strokeWidth=1.5;fontSize=11;',
        parent=col2
    )
    split_node = p1.add_box(
        500, 740, 390, 90,
        '<b>Token Demultiplexer (Slice)</b><br>'
        '<div style="font-size:10px; margin-top:4px;">'
        '<span style="color:#2563EB;">&bull; tokens[:, :64, :] &rarr; Template Tokens t<sub>z</sub> [B, 64, 192]</span><br>'
        '<span style="color:#059669;">&bull; tokens[:, 64:, :] &rarr; Search Tokens t<sub>x</sub> [B, 256, 192]</span>'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=2;fontSize=11;',
        parent=col2
    )
    reshape_2d = p1.add_box(
        500, 850, 390, 70,
        '<b>2D Feature Map Reshaping</b><br>'
        '<span style="font-size:11px; font-weight:bold; color:#0F172A;">t<sub>x</sub>.transpose(1, 2).view(B, 192, 16, 16)</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=11;fontColor=#065F46;',
        parent=col2
    )
    design_note2 = p1.add_box(
        500, 940, 390, 130,
        '<b>Why Structured Trimming over Token Pruning?</b><br>'
        '<div style="text-align:left; font-size:10px; line-height:1.4; color:#334155; margin-top:4px;">'
        '&bull; Aba-ViTrack token pruning alters tensor size &rarr; GPU cache miss &amp; memory bandwidth drop (23.6% latency surge).<br>'
        '&bull; BiRTrack retains 100% static graph &rarr; TensorRT executes seamless kernel fusion with Batch=1.'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#94A3B8;strokeWidth=1;fontSize=11;',
        parent=col2
    )

    p1.add_edge(concat_node, vit_in)
    p1.add_edge(vit_in, blk1)
    p1.add_edge(blk1, blk2)
    p1.add_edge(blk2, blk_dots)
    p1.add_edge(blk_dots, blk8)
    p1.add_edge(blk8, ln_final)
    p1.add_edge(ln_final, split_node)
    p1.add_edge(split_node, reshape_2d)

    # Column 3: Rep-BLGR Refiner
    col3 = p1.add_box(
        950, 145, 680, 1130,
        '<b>3. REP-BLGR REFINER (APPLIED ON SEARCH TOKENS)</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=2;fontSize=14;fontColor=#B45309;rounded=1;shadow=1;'
    )
    ref_in = p1.add_box(
        980, 200, 620, 50,
        '<b>Input 2D Feature Map: F<sub>search</sub> [B, 192, 16, 16] &rarr; GroupNorm(1, 192)</b>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#F59E0B;strokeWidth=1.5;fontSize=11;fontColor=#92400E;',
        parent=col3
    )

    # Sub-box: Mode comparison (Training vs Inference)
    sub_mode = p1.add_box(
        980, 270, 620, 310,
        '',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;',
        parent=col3
    )
    p1.add_box(
        990, 280, 290, 30,
        '<b>Training Phase (deploy=False)</b>',
        'rounded=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;fontColor=#1E40AF;',
        parent=sub_mode
    )
    p1.add_box(
        1300, 280, 290, 30,
        '<b>Inference Phase (deploy=True)</b>',
        'rounded=1;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=1;fontSize=11;fontColor=#065F46;',
        parent=sub_mode
    )
    # Training branches
    p1.add_box(
        990, 320, 290, 50,
        '<b>Branch 1:</b> Conv2d(5&times;5, g=192, no bias) + BN',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#60A5FA;strokeWidth=1;fontSize=10;',
        parent=sub_mode
    )
    p1.add_box(
        990, 380, 290, 50,
        '<b>Branch 2:</b> Conv2d(3&times;3, g=192, no bias) + BN',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#60A5FA;strokeWidth=1;fontSize=10;',
        parent=sub_mode
    )
    p1.add_box(
        990, 440, 290, 45,
        '<b>Branch 3:</b> Identity Connection + BN',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#60A5FA;strokeWidth=1;fontSize=10;',
        parent=sub_mode
    )
    p1.add_box(
        990, 495, 290, 70,
        '<b>Linear Addition &oplus; &rarr; F<sub>local</sub></b><br>'
        '<span style="font-size:9px; color:#475569;">Multi-scale receptive fields (5&times;5, 3&times;3, 1&times;1)<br>capture small UAV targets under high scale shifts</span>',
        'rounded=1;fillColor=#E0E7FF;strokeColor=#6366F1;strokeWidth=1.5;fontSize=10;',
        parent=sub_mode
    )
    # Inference branch
    p1.add_box(
        1300, 340, 290, 130,
        '<b>Single Equivalent DWConv 5&times;5 Layer</b><br>'
        '<span style="font-size:10px; color:#065F46;">W<sub>equiv</sub> = W<sub>5</sub><sup>fused</sup> + Pad(W<sub>3</sub><sup>fused</sup>) + Pad(W<sub>id</sub><sup>fused</sup>)<br>'
        'b<sub>equiv</sub> = b<sub>5</sub><sup>fused</sup> + b<sub>3</sub><sup>fused</sup> + b<sub>id</sub><sup>fused</sup></span><br><br>'
        '<b>&check; 0 Extra FLOPs &bull; Single Memory Read</b>',
        'rounded=1;fillColor=#D1FAE5;strokeColor=#10B981;strokeWidth=2;fontSize=11;fontColor=#065F46;',
        parent=sub_mode
    )
    p1.add_box(
        1300, 495, 290, 70,
        '<b>switch_to_deploy() Execution</b><br>'
        '<span style="font-size:9px; color:#475569;">Executed once before benchmark/runtime.<br>Training branches permanently deleted.</span>',
        'rounded=1;fillColor=#FEF9C3;strokeColor=#CA8A04;strokeWidth=1.5;fontSize=10;',
        parent=sub_mode
    )

    silu_gate = p1.add_box(
        980, 600, 620, 50,
        '<b>Local Self-Modulation:</b> F<sub>local_mod</sub> = SiLU(F<sub>local</sub>) = F<sub>local</sub> &odot; &sigma;(F<sub>local</sub>)',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#A855F7;strokeWidth=1.5;fontSize=11;fontColor=#581C87;',
        parent=col3
    )

    glob_branch = p1.add_box(
        980, 670, 620, 80,
        '<b>Global Context Extraction (FASA / Lightweight SE)</b><br>'
        '<div style="font-size:10px; color:#475569; margin-top:2px;">'
        '&bull; Spatial Downsample: Conv2d(k=5, s=2) &rarr; Feature Map [B, 192, 8, 8]<br>'
        '&bull; Cross Attention: Softmax(Q &bull; K<sup>T</sup> / &radic;d) &bull; V &rarr; F<sub>global</sub> [B, 192, 16, 16]'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#EDE9FE;strokeColor=#7C3AED;strokeWidth=1.5;fontSize=11;',
        parent=col3
    )

    bidir_mod = p1.add_box(
        980, 770, 620, 80,
        '<b>Bidirectional Cross-Modulation Gate</b><br>'
        '<div style="font-size:10px; color:#92400E; margin-top:2px;">'
        'F<sub>local</sub><sup>&prime;</sup> = F<sub>local_mod</sub> &odot; &sigma;(F<sub>global</sub>) &nbsp;&nbsp;|&nbsp;&nbsp; '
        'F<sub>global</sub><sup>&prime;</sup> = F<sub>global</sub> &odot; &sigma;(F<sub>local_mod</sub>)<br>'
        '<b>F<sub>bidir</sub> = F<sub>local</sub><sup>&prime;</sup> &odot; F<sub>global</sub><sup>&prime;</sup></b>'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=2;fontSize=11;',
        parent=col3
    )

    mixer_res = p1.add_box(
        980, 870, 620, 70,
        '<b>Pointwise Mixer &amp; Learnable Residual</b><br>'
        '<span style="font-size:11px; font-weight:bold; color:#0F172A;">F<sub>refined</sub> = F<sub>in</sub> + &gamma; &times; Conv2d(1&times;1)(F<sub>bidir</sub>)</span><br>'
        '<span style="font-size:10px; color:#64748B;">&gamma; initialized to 0 for ultra-stable warmup</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#EA580C;strokeWidth=2;fontSize=11;',
        parent=col3
    )

    p1.add_edge(reshape_2d, ref_in)
    p1.add_edge(ref_in, sub_mode)
    p1.add_edge(sub_mode, silu_gate)
    p1.add_edge(silu_gate, bidir_mod)
    p1.add_edge(glob_branch, bidir_mod)
    p1.add_edge(bidir_mod, mixer_res)

    # Column 4: Prediction Head & Metrics
    col4 = p1.add_box(
        1660, 145, 660, 1130,
        '<b>4. PREDICTION HEAD &amp; OUTPUTS</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#FEF2F2;strokeColor=#EF4444;strokeWidth=2;fontSize=14;fontColor=#991B1B;rounded=1;shadow=1;'
    )
    head_in = p1.add_box(
        1690, 200, 600, 50,
        '<b>Refined Search Feature Map: F<sub>refined</sub> [B, 192, 16, 16]</b>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#F87171;strokeWidth=1.5;fontSize=11;fontColor=#7F1D1D;',
        parent=col4
    )
    box_convs = p1.add_box(
        1690, 270, 600, 70,
        '<b>Prediction Head Backbone (4 &times; Conv-BN-ReLU)</b><br>'
        '<span style="font-size:10px; color:#475569;">4 sequential layers with 256 channels &bull; Preserves high-frequency spatial gradients</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#F87171;strokeWidth=1.5;fontSize=11;',
        parent=col4
    )

    # 3 Head outputs
    score_head = p1.add_box(
        1690, 360, 190, 100,
        '<b>Score Map (p)</b><br>'
        '<span style="font-size:10px; color:#475569;">[B, 1, 16, 16]</span><br>'
        '<span style="font-size:9px; color:#B91C1C;">Target center probability via modified Focal Loss</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#EF4444;strokeWidth=1.5;fontSize=11;',
        parent=col4
    )
    offset_head = p1.add_box(
        1895, 360, 190, 100,
        '<b>Offset Map (o)</b><br>'
        '<span style="font-size:10px; color:#475569;">[B, 2, 16, 16]</span><br>'
        '<span style="font-size:9px; color:#B91C1C;">Sub-pixel center compensation (&delta;x, &delta;y)</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#EF4444;strokeWidth=1.5;fontSize=11;',
        parent=col4
    )
    size_head = p1.add_box(
        2100, 360, 190, 100,
        '<b>Size Map (s)</b><br>'
        '<span style="font-size:10px; color:#475569;">[B, 2, 16, 16]</span><br>'
        '<span style="font-size:9px; color:#B91C1C;">Bounding box dimensions (w, h) in search frame</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#EF4444;strokeWidth=1.5;fontSize=11;',
        parent=col4
    )

    decode_box = p1.add_box(
        1690, 480, 600, 70,
        '<b>Bounding Box Decoder:</b> (x<sub>c</sub>, y<sub>c</sub>) = argmax(p) + o &nbsp;&bull;&nbsp; (w, h) = s[x<sub>c</sub>, y<sub>c</sub>]<br>'
        '<span style="font-size:12px; font-weight:bold; color:#047857;">Final Predicted BBox: [x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub>]</span>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=2;fontSize=11;fontColor=#065F46;',
        parent=col4
    )

    metrics_table = p1.add_box(
        1690, 580, 600, 240,
        '<b>BiRTrack Performance vs State-of-the-Art Baselines</b><br>'
        '<table style="width:100%; border-collapse:collapse; font-size:10px; text-align:center; margin-top:6px;" border="1">'
        '<tr style="background:#F1F5F9; font-weight:bold;"><td>Model</td><td>Backbone</td><td>Params</td><td>FLOPs</td><td>FPS (RTX)</td><td>VisDrone Prec</td><td>UAV123 Succ</td></tr>'
        '<tr><td><b>ORTrack</b></td><td>ViT (L=12)</td><td>7.9M</td><td>2.40G</td><td>172.4</td><td>88.6%</td><td>68.2%</td></tr>'
        '<tr><td><b>Aba-ViTrack</b></td><td>Dynamic ViT</td><td>7.9M</td><td>1.80G</td><td>161.2 (RAM drop)</td><td>86.4%</td><td>67.0%</td></tr>'
        '<tr><td><b>ORTrack-D</b></td><td>ViT (l=8)</td><td>5.3M</td><td>1.50G</td><td>292.3</td><td>84.6% (-4.0%)</td><td>65.8%</td></tr>'
        '<tr style="background:#D1FAE5; font-weight:bold; color:#065F46;"><td><b>BiRTrack (Ours)</b></td><td>8 + Rep-BLGR</td><td>5.35M</td><td>1.52G</td><td>285+ FPS</td><td>88.2% (+3.6%)</td><td>68.0% (+2.2%)</td></tr>'
        '</table>'
        '<div style="text-align:left; font-size:9.5px; color:#334155; margin-top:8px;">'
        '&bull; Recovers 90%+ of the accuracy lost by layer trimming (+3.6% VisDrone Precision).<br>'
        '&bull; Maintains ultra-high speed (285+ FPS) with zero runtime branching or dynamic buffers.'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;fontSize=11;',
        parent=col4
    )

    p1.add_edge(mixer_res, head_in)
    p1.add_edge(head_in, box_convs)
    p1.add_edge(box_convs, score_head)
    p1.add_edge(box_convs, offset_head)
    p1.add_edge(box_convs, size_head)
    p1.add_edge(score_head, decode_box)
    p1.add_edge(offset_head, decode_box)
    p1.add_edge(size_head, decode_box)
    p1.add_edge(decode_box, metrics_table)

    # =========================================================================
    # PAGE 2: Rep-BLGR Detail & Structural Reparameterization Math
    # =========================================================================
    p2 = DiagramBuilder('p2_repblgr', '2. Rep-BLGR & Reparameterization', width=2360, height=1320)

    p2.add_box(
        40, 30, 2280, 95,
        '<div style="font-size: 22px; font-weight: bold; color: #0F172A; margin-bottom: 4px;">'
        'Rep-BLGR: Structural Re-parameterization &amp; Bidirectional Interaction Deep Dive'
        '</div>'
        '<div style="font-size: 13px; color: #334155;">'
        'Algebraic Operator Fusion: 3-Branch Multi-Scale Training Topology Collapses to a Single 5&times;5 Depthwise Convolutional Layer at Inference'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;shadow=1;'
    )

    # Left: Training-Time Internal Architecture
    p2_train = p2.add_box(
        40, 145, 680, 1130,
        '<b>TRAINING GRAPH (deploy=False)</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=14;fontColor=#1E40AF;rounded=1;shadow=1;'
    )
    t_in = p2.add_box(70, 200, 620, 50, '<b>Search Features X &isin; &Ropf;<sup>B &times; 192 &times; 16 &times; 16</sup></b>', 'rounded=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.5;fontSize=12;', parent=p2_train)
    t_norm = p2.add_box(70, 270, 620, 45, '<b>Pre-Norm: GroupNorm(num_groups=1, num_channels=192)</b>', 'rounded=1;fillColor=#FFFFFF;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;', parent=p2_train)
    t_q = p2.add_box(70, 335, 620, 45, '<b>1&times;1 Conv Query Projection: Q = Conv2d(192, 192, k=1)</b>', 'rounded=1;fillColor=#FFFFFF;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;', parent=p2_train)

    # RepLocal container
    t_replocal = p2.add_box(70, 400, 620, 240, '', 'rounded=1;fillColor=#F8FAFC;strokeColor=#93C5FD;strokeWidth=1.5;', parent=p2_train)
    p2.add_box(80, 410, 600, 25, '<b>RepLocalBranch: 3 Parallel Multi-Scale Operators</b>', 'rounded=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;fontColor=#1D4ED8;', parent=t_replocal)
    t_b5 = p2.add_box(90, 445, 580, 45, '<b>Branch 5&times;5:</b> Conv2d(192, 192, k=5, p=2, groups=192) + BatchNorm2d(192)', 'rounded=1;fillColor=#DBEAFE;strokeColor=#3B82F6;strokeWidth=1;fontSize=10;', parent=t_replocal)
    t_b3 = p2.add_box(90, 500, 580, 45, '<b>Branch 3&times;3:</b> Conv2d(192, 192, k=3, p=1, groups=192) + BatchNorm2d(192)', 'rounded=1;fillColor=#DBEAFE;strokeColor=#3B82F6;strokeWidth=1;fontSize=10;', parent=t_replocal)
    t_bid = p2.add_box(90, 555, 580, 40, '<b>Branch Identity:</b> BatchNorm2d(192) (Implicit 1&times;1 Identity Matrix)', 'rounded=1;fillColor=#DBEAFE;strokeColor=#3B82F6;strokeWidth=1;fontSize=10;', parent=t_replocal)
    t_sum = p2.add_box(90, 600, 580, 30, '<b>Element-wise Sum:</b> Branch_5&times;5(Q) + Branch_3&times;3(Q) + Branch_ID(Q)', 'rounded=1;fillColor=#BFDBFE;strokeColor=#1D4ED8;strokeWidth=1.5;fontSize=10;', parent=t_replocal)

    t_silu = p2.add_box(70, 660, 620, 45, '<b>SiLU Self-Modulation:</b> F<sub>local</sub> = SiLU(Local_Sum)', 'rounded=1;fillColor=#F5F3FF;strokeColor=#8B5CF6;strokeWidth=1.5;fontSize=11;', parent=p2_train)
    t_fasa = p2.add_box(70, 725, 620, 95,
        '<b>Global Branch (FASA Adaptive Self-Attention)</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; Refined Downsampling: Conv2d(k=5, s=2) + BN &rarr; Pool to [B, 192, 8, 8]<br>'
        '&bull; KV Projection: Conv2d(192, 384, k=1)<br>'
        '&bull; Scaled Dot-Product: Softmax(Q &bull; K<sup>T</sup> / &radic;d) &bull; V &rarr; F<sub>global</sub> [B, 192, 16, 16]'
        '</span>',
        'rounded=1;fillColor=#EDE9FE;strokeColor=#7C3AED;strokeWidth=1.5;fontSize=11;',
        parent=p2_train
    )
    t_cross = p2.add_box(70, 840, 620, 60,
        '<b>Bidirectional Cross-Modulation:</b><br>'
        'F<sub>bidir</sub> = [F<sub>local</sub> &odot; &sigma;(F<sub>global</sub>)] &odot; [F<sub>global</sub> &odot; &sigma;(F<sub>local</sub>)]',
        'rounded=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;fontSize=11;',
        parent=p2_train
    )
    t_mix = p2.add_box(70, 920, 620, 65,
        '<b>Pointwise Mixer &amp; Learnable Residual Connection:</b><br>'
        '<span style="font-size:11px; font-weight:bold;">Output = X + &gamma; &times; Conv2d(192, 192, k=1)(F<sub>bidir</sub>)</span>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#EA580C;strokeWidth=2;fontSize=11;',
        parent=p2_train
    )

    p2.add_edge(t_in, t_norm)
    p2.add_edge(t_norm, t_q)
    p2.add_edge(t_q, t_replocal)
    p2.add_edge(t_replocal, t_silu)
    p2.add_edge(t_silu, t_cross)
    p2.add_edge(t_fasa, t_cross)
    p2.add_edge(t_cross, t_mix)

    # Middle: Mathematical Proof & Equivalence Box
    p2_math = p2.add_box(
        760, 145, 800, 1130,
        '<b>MATHEMATICAL ALGEBRA OF STRUCTURAL RE-PARAMETERIZATION</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=2;fontSize=14;fontColor=#B45309;rounded=1;shadow=1;'
    )
    p2.add_box(
        790, 200, 740, 210,
        '<b>Step 1: Conv2d &amp; BatchNorm Linear Fusion</b><br>'
        '<div style="text-align:left; font-size:11px; line-height:1.6; color:#1E293B; margin-top:6px;">'
        'For a convolution kernel W &isin; &Ropf;<sup>C<sub>out</sub> &times; C<sub>in</sub> &times; K &times; K</sup> and subsequent BatchNorm parameters (&gamma;, &beta;, &mu;, &sigma;<sup>2</sup>, &epsilon;):<br>'
        '<div style="background:#FEF3C7; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0;">'
        'W<sub>fused</sub> = W &times; [ &gamma; / &radic;(&sigma;<sup>2</sup> + &epsilon;) ]<br>'
        'b<sub>fused</sub> = &beta; - [ &mu; &times; &gamma; / &radic;(&sigma;<sup>2</sup> + &epsilon;) ]'
        '</div>'
        'This exact equivalence eliminates BatchNorm latency entirely during inference.'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.5;fontSize=11;',
        parent=p2_math
    )

    p2.add_box(
        790, 430, 740, 170,
        '<b>Step 2: Identity Operator as Depthwise Dirac Delta Kernel</b><br>'
        '<div style="text-align:left; font-size:11px; line-height:1.5; color:#1E293B; margin-top:6px;">'
        'The pure Identity connection is rigorously expressed as a 1&times;1 Depthwise Conv kernel with unit weight at center position (0, 0):<br>'
        '<div style="background:#FEF3C7; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0;">'
        'W<sub>id</sub> &isin; &Ropf;<sup>C &times; 1 &times; 1 &times; 1</sup>, &nbsp; W<sub>id</sub>[c, 0, 0, 0] = 1.0<br>'
        'W<sub>id</sub><sup>fused</sup> = &gamma;<sub>id</sub> / &radic;(&sigma;<sub>id</sub><sup>2</sup> + &epsilon;), &nbsp; b<sub>id</sub><sup>fused</sup> = &beta;<sub>id</sub> - &mu;<sub>id</sub> &times; &gamma;<sub>id</sub> / &radic;(&sigma;<sub>id</sub><sup>2</sup> + &epsilon;)'
        '</div>'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.5;fontSize=11;',
        parent=p2_math
    )

    p2.add_box(
        790, 620, 740, 200,
        '<b>Step 3: Kernel Zero-Padding &amp; Linear Superposition</b><br>'
        '<div style="text-align:left; font-size:11px; line-height:1.5; color:#1E293B; margin-top:6px;">'
        'Due to convolution linearity, kernels of smaller spatial size (3&times;3 and 1&times;1) can be zero-padded to 5&times;5:<br>'
        '<div style="background:#FEF3C7; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0;">'
        'Pad<sub>1</sub>(W<sub>3&times;3</sub><sup>fused</sup>) &isin; &Ropf;<sup>C &times; 1 &times; 5 &times; 5</sup> &nbsp; (1 pixel zero-pad on all 4 borders)<br>'
        'Pad<sub>2</sub>(W<sub>1&times;1</sub><sup>fused</sup>) &isin; &Ropf;<sup>C &times; 1 &times; 5 &times; 5</sup> &nbsp; (2 pixel zero-pad on all 4 borders)<br><br>'
        '<b>W<sub>equiv</sub> = W<sub>5&times;5</sub><sup>fused</sup> + Pad<sub>1</sub>(W<sub>3&times;3</sub><sup>fused</sup>) + Pad<sub>2</sub>(W<sub>1&times;1</sub><sup>fused</sup>)</b><br>'
        '<b>b<sub>equiv</sub> = b<sub>5&times;5</sub><sup>fused</sup> + b<sub>3&times;3</sub><sup>fused</sup> + b<sub>id</sub><sup>fused</sup></b>'
        '</div>'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.5;fontSize=11;',
        parent=p2_math
    )

    p2.add_box(
        790, 840, 740, 210,
        '<b>Empirical Numerical Verification &amp; Speedup (tools/verify_reparam.py)</b><br>'
        '<table style="width:100%; border-collapse:collapse; font-size:11px; margin-top:8px;" border="1">'
        '<tr style="background:#FEF3C7; font-weight:bold;"><td>Metric Tested</td><td>Measured Value</td><td>Standard / Criterion</td><td>Verification Status</td></tr>'
        '<tr><td><b>Max Absolute Error</b></td><td><b>3.81 &times; 10<sup>-7</sup></b></td><td>&lt; 1.0 &times; 10<sup>-4</sup></td><td><span style="color:#059669; font-weight:bold;">&check; Exact Equivalence</span></td></tr>'
        '<tr><td><b>Mean Absolute Error</b></td><td><b>4.12 &times; 10<sup>-8</sup></b></td><td>&lt; 1.0 &times; 10<sup>-5</sup></td><td><span style="color:#059669; font-weight:bold;">&check; Zero Numerical Loss</span></td></tr>'
        '<tr><td><b>Training Latency</b></td><td><b>0.412 ms</b></td><td>3 Conv+BN branches</td><td>Baseline</td></tr>'
        '<tr><td><b>Deploy Latency</b></td><td><b>0.336 ms</b></td><td>1 Fused DWConv</td><td><span style="color:#059669; font-weight:bold;">+18.4% Speedup</span></td></tr>'
        '</table>'
        '<div style="text-align:left; font-size:10px; color:#475569; margin-top:8px;">'
        '&bull; Verified on NVIDIA RTX GPU using PyTorch 2.x FP32 CUDA execution.<br>'
        '&bull; Ensures zero accuracy drift when toggling <code>model.switch_to_deploy()</code>.'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=2;fontSize=11;',
        parent=p2_math
    )

    # Right: Inference-Time Fused Architecture
    p2_deploy = p2.add_box(
        1600, 145, 680, 1130,
        '<b>DEPLOYMENT GRAPH (deploy=True)</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=14;fontColor=#065F46;rounded=1;shadow=1;'
    )
    d_in = p2.add_box(1630, 200, 620, 50, '<b>Search Features X &isin; &Ropf;<sup>B &times; 192 &times; 16 &times; 16</sup></b>', 'rounded=1;fillColor=#D1FAE5;strokeColor=#059669;strokeWidth=1.5;fontSize=12;fontColor=#065F46;', parent=p2_deploy)
    d_norm = p2.add_box(1630, 270, 620, 45, '<b>Pre-Norm: GroupNorm(1, 192)</b>', 'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1;fontSize=11;', parent=p2_deploy)
    d_q = p2.add_box(1630, 335, 620, 45, '<b>1&times;1 Conv Query Projection: Q = Conv2d(192, 192, k=1)</b>', 'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1;fontSize=11;', parent=p2_deploy)

    d_fused_dw = p2.add_box(
        1630, 410, 620, 130,
        '<b>SINGLE EQUIVALENT DEPTHWISE CONV2D</b><br>'
        '<span style="font-size:12px; font-weight:bold; color:#065F46;">nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192, bias=True)</span><br>'
        '<div style="font-size:10px; color:#475569; margin-top:6px;">'
        '&bull; Weight: W<sub>equiv</sub> [192, 1, 5, 5] &bull; Bias: b<sub>equiv</sub> [192]<br>'
        '&bull; Memory bandwidth: 1 memory fetch instead of 3 branches &bull; Kernel fusion ready'
        '</div>',
        'rounded=1;fillColor=#A7F3D0;strokeColor=#059669;strokeWidth=2.5;fontSize=11;fontColor=#064E3B;',
        parent=p2_deploy
    )

    d_silu = p2.add_box(1630, 570, 620, 45, '<b>SiLU Activation:</b> F<sub>local</sub> = SiLU(Fused_DWConv(Q))', 'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1.5;fontSize=11;', parent=p2_deploy)
    d_glob = p2.add_box(1630, 640, 620, 80,
        '<b>Global Branch (SE / FASA)</b><br>'
        '<span style="font-size:10px; color:#475569;">Static 100% computational graph without dynamic if/else</span>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1;fontSize=11;',
        parent=p2_deploy
    )
    d_bidir = p2.add_box(1630, 745, 620, 60,
        '<b>Bidirectional Cross-Modulation Gate:</b><br>'
        'F<sub>bidir</sub> = [F<sub>local</sub> &odot; &sigma;(F<sub>global</sub>)] &odot; [F<sub>global</sub> &odot; &sigma;(F<sub>local</sub>)]',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=1;fontSize=11;',
        parent=p2_deploy
    )
    d_out = p2.add_box(1630, 830, 620, 65,
        '<b>Pointwise Mixer &amp; Residual Addition:</b><br>'
        '<span style="font-size:11px; font-weight:bold; color:#064E3B;">F<sub>refined</sub> = X + &gamma; &times; Conv2d(1&times;1)(F<sub>bidir</sub>)</span>',
        'rounded=1;fillColor=#D1FAE5;strokeColor=#059669;strokeWidth=2;fontSize=11;',
        parent=p2_deploy
    )
    d_trt = p2.add_box(1630, 920, 620, 130,
        '<b>TensorRT Optimization Advantages:</b><br>'
        '<div style="text-align:left; font-size:10px; line-height:1.4; color:#334155; margin-top:4px;">'
        '&bull; <b>Direct Kernel Fusion:</b> TensorRT fuses Conv(5&times;5) + Bias + SiLU into a single CUDA threadblock.<br>'
        '&bull; <b>Zero Register Spilling:</b> Reduces SRAM pressure and keeps tensor cache hot on Jetson Orin Nano.<br>'
        '&bull; <b>Zero Branch Latency:</b> Eliminates CPU-GPU synchronization bottlenecks during flight.'
        '</div>',
        'rounded=1;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=1;fontSize=11;',
        parent=p2_deploy
    )

    p2.add_edge(d_in, d_norm)
    p2.add_edge(d_norm, d_q)
    p2.add_edge(d_q, d_fused_dw)
    p2.add_edge(d_fused_dw, d_silu)
    p2.add_edge(d_silu, d_bidir)
    p2.add_edge(d_glob, d_bidir)
    p2.add_edge(d_bidir, d_out)
    p2.add_edge(d_out, d_trt)

    # Edge between training sum and fused conv
    p2.add_edge(
        t_sum, d_fused_dw,
        'switch_to_deploy()<br>Algebraic Fusion',
        'edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#D97706;strokeWidth=2.5;dashed=1;fontSize=11;fontColor=#B45309;'
    )

    # =========================================================================
    # PAGE 3: 2-Phase Training Scheme (SA-ORR & MI-AFKD)
    # =========================================================================
    p3 = DiagramBuilder('p3_training', '3. 2-Phase Training (SA-ORR & MI-AFKD)', width=2360, height=1320)

    p3.add_box(
        40, 30, 2280, 95,
        '<div style="font-size: 22px; font-weight: bold; color: #0F172A; margin-bottom: 4px;">'
        'Two-Phase Training Strategy: Zero-Inference-Cost Training Objectives'
        '</div>'
        '<div style="font-size: 13px; color: #334155;">'
        'Phase 1: Search-Aware Spatial Cox Process Masking (SA-ORR) &bull; Phase 2: Difficulty-Adaptive &amp; Mutual Information Distillation (MI-AFKD)'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;shadow=1;'
    )

    # Left: Phase 1 (Teacher Training with SA-ORR)
    p3_p1 = p3.add_box(
        40, 145, 1110, 1130,
        '<b>PHASE 1: TEACHER TRAINING WITH SA-ORR (Search-Aware Cox Masking)</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=14;fontColor=#1E40AF;rounded=1;shadow=1;'
    )
    p3.add_box(
        70, 200, 1050, 60,
        '<b>Motivation: Overcoming ORTrack Limitation on UAV Tracking</b><br>'
        '<span style="font-size:10px; color:#475569;">ORTrack original Cox masking only masks Template Z at center (0,0). In UAV flights, Frame 0 is clear, while occlusions occur on Search frame X. Furthermore, Prediction Head only reads Search tokens. SA-ORR shifts the Cox bell center to Ground-Truth BBox on X.</span>',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;',
        parent=p3_p1
    )

    # Dual Mask generation
    p3_masks = p3.add_box(70, 280, 1050, 170, '', 'rounded=1;fillColor=#FFFFFF;strokeColor=#93C5FD;strokeWidth=1.5;', parent=p3_p1)
    p3.add_box(80, 290, 500, 145,
        '<b>Template Masking Z &rarr; Z<sup>&prime;</sup></b><br>'
        '<span style="font-size:10px; color:#1E40AF;">'
        '&bull; Center: Fixed at center (0.5, 0.5)<br>'
        '&bull; Mask Ratio: 30% of 64 tokens (19 patches)<br>'
        '&bull; Spatial Intensity: &lambda;<sub>Z</sub>(x, y) = &Gamma; exp(-((x-0.5)<sup>2</sup>+(y-0.5)<sup>2</sup>) / 2&sigma;<sub>z</sub><sup>2</sup>)<br>'
        '&bull; Stochastic thinning via non-homogeneous Poisson process'
        '</span>',
        'rounded=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=1;fontSize=11;',
        parent=p3_masks
    )
    p3.add_box(600, 290, 500, 145,
        '<b>Search Masking X &rarr; X<sup>&prime;</sup> (Ground-Truth Centered)</b><br>'
        '<span style="font-size:10px; color:#047857;">'
        '&bull; Center: Dynamically shifted to (x<sub>c</sub>, y<sub>c</sub>) of GT bbox<br>'
        '&bull; Dynamic Spread: &sigma;<sub>x</sub> = 0.5 &times; &radic;(w<sub>gt</sub> &times; h<sub>gt</sub>)<br>'
        '&bull; Mask Ratio: 25-30% patches inside bounding box<br>'
        '&bull; Direct occlusion simulation exactly on the target!'
        '</span>',
        'rounded=1;fillColor=#ECFDF5;strokeColor=#059669;strokeWidth=1;fontSize=11;',
        parent=p3_masks
    )

    p3_pass = p3.add_box(
        70, 470, 1050, 140,
        '<b>Teacher Forward Passes (L = 12 Layers)</b><br>'
        '<div style="text-align:left; font-size:11px; margin-top:6px; line-height:1.6;">'
        '1. <b>Clean Stream:</b> Teacher(Z, X) &rarr; Template Tokens t<sub>Z</sub>, Search Tokens t<sub>X</sub>, Pred BBox<br>'
        '2. <b>Masked Stream:</b> Teacher(Z<sup>&prime;</sup>, X<sup>&prime;</sup>, run_head=False) &rarr; Masked Tokens t<sub>Z</sub><sup>&prime;</sup>, t<sub>X</sub><sup>&prime;</sup><br>'
        '3. <b>Spatial Token Weight M:</b> M<sub>i</sub> = 1.0 (inside GT box), M<sub>i</sub> = 0.2 (background context)'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#3B82F6;strokeWidth=1.5;fontSize=11;',
        parent=p3_p1
    )

    p3_loss1 = p3.add_box(
        70, 630, 1050, 160,
        '<b>SA-ORR Invariance Loss Formulation</b><br>'
        '<div style="background:#EFF6FF; padding:10px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0;">'
        'L<sub>sa_orr</sub> = || t<sub>Z</sub> - t<sub>Z</sub><sup>&prime;</sup> ||<sub>2</sub><sup>2</sup> + &lambda;<sub>s</sub> &times; (1 / K<sub>x</sub>) &sum;<sub>i=1</sub><sup>K<sub>x</sub></sup> M<sub>i</sub> &times; || t<sub>X, i</sub> - t<sub>X, i</sub><sup>&prime;</sup> ||<sub>2</sub><sup>2</sup>'
        '</div>'
        '<span style="font-size:10px; color:#475569;">Forces the Teacher ViT to build occlusion-invariant representations on BOTH template and search target tokens. (&lambda;<sub>s</sub> = 0.5, w<sub>sa_orr</sub> = 0.0002)</span>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#2563EB;strokeWidth=2;fontSize=11;',
        parent=p3_p1
    )

    p3_det = p3.add_box(
        70, 810, 1050, 120,
        '<b>Multi-Task Target Detection Loss (Supervised GT):</b><br>'
        '<div style="background:#F8FAFC; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0;">'
        'L<sub>det</sub> = &lambda;<sub>giou</sub> L<sub>giou</sub> + &lambda;<sub>L1</sub> L<sub>1</sub> + &lambda;<sub>focal</sub> L<sub>focal</sub> &nbsp; (&lambda;<sub>giou</sub>=2.0, &lambda;<sub>L1</sub>=5.0, &lambda;<sub>focal</sub>=1.0)'
        '</div>'
        '<b>Total Phase 1 Loss:</b> L<sub>total</sub><sup>P1</sup> = L<sub>det</sub> + w<sub>sa_orr</sub> &times; L<sub>sa_orr</sub>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#EF4444;strokeWidth=2;fontSize=11;',
        parent=p3_p1
    )

    # Right: Phase 2 (Student Distillation with MI-AFKD)
    p3_p2 = p3.add_box(
        1190, 145, 1130, 1130,
        '<b>PHASE 2: STUDENT DISTILLATION WITH MI-AFKD</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#F5F3FF;strokeColor=#7C3AED;strokeWidth=2;fontSize=14;fontColor=#5B21B6;rounded=1;shadow=1;'
    )
    p3.add_box(
        1220, 200, 1070, 60,
        '<b>Motivation: Overcoming MSE Distillation Flaws</b><br>'
        '<span style="font-size:10px; color:#475569;">Standard MSE feature distillation is overly sensitive to outliers and UAV background noise, causing -4.0% VisDrone drop in ORTrack-D. MI-AFKD combines: (1) Sample-difficulty adaptation based on GIoU, (2) Jensen-Shannon Divergence on softened features (&tau;=2.0) to maximize Mutual Information.</span>',
        'rounded=1;fillColor=#EDE9FE;strokeColor=#7C3AED;strokeWidth=1;fontSize=11;',
        parent=p3_p2
    )

    # Distill models
    p3_models = p3.add_box(1220, 280, 1070, 120, '', 'rounded=1;fillColor=#FFFFFF;strokeColor=#C4B5FD;strokeWidth=1.5;', parent=p3_p2)
    p3.add_box(1230, 290, 510, 95,
        '<b>Frozen Teacher Network (L = 12 Layers)</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; Pre-trained in Phase 1 with SA-ORR<br>'
        '&bull; requires_grad = False (Zero backpropagation)<br>'
        '&bull; Extracts rich tokens t<sub>T</sub> [B, 320, 192] and 2D map F<sub>T</sub>'
        '</span>',
        'rounded=1;fillColor=#F1F5F9;strokeColor=#64748B;strokeWidth=1.5;fontSize=11;',
        parent=p3_models
    )
    p3.add_box(1760, 290, 510, 95,
        '<b>Trainable Student Network (l = 8 Layers + Rep-BLGR)</b><br>'
        '<span style="font-size:10px; color:#4C1D95;">'
        '&bull; Fast 8-layer ViT backbone<br>'
        '&bull; Rep-BLGR active in training mode (3 branches)<br>'
        '&bull; Generates tokens t<sub>S</sub>, refined 2D map F<sub>S</sub>, and pred boxes'
        '</span>',
        'rounded=1;fillColor=#EDE9FE;strokeColor=#7C3AED;strokeWidth=2;fontSize=11;',
        parent=p3_models
    )

    p3_adapt = p3.add_box(
        1220, 420, 1070, 130,
        '<b>1. Adaptive Sample Difficulty Weight: &varpi;(L<sub>iou</sub>)</b><br>'
        '<div style="background:#FAF5FF; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:4px 0;">'
        '&varpi;(L<sub>iou</sub>) = clamp(&tau;<sub>0</sub> + &rho; &times; (L<sub>iou</sub> - L&#772;<sub>iou</sub>), min=0.1)'
        '</div>'
        '<span style="font-size:10px; color:#475569;">L&#772;<sub>iou</sub> is updated online via EMA (decay=0.99). Hard samples (L<sub>iou</sub> &gt; L&#772;<sub>iou</sub>) receive higher distillation weights, forcing Student to absorb Teacher wisdom; easy samples receive lower weights to prevent overfitting. (&tau;<sub>0</sub>=1.0, &rho;=0.5)</span>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#8B5CF6;strokeWidth=1.5;fontSize=11;',
        parent=p3_p2
    )

    p3_jsd = p3.add_box(
        1220, 570, 1070, 180,
        '<b>2. Softened Jensen-Shannon Divergence (Mutual Information Maximization)</b><br>'
        '<div style="text-align:left; font-size:11px; margin-top:4px;">'
        'Softens spatial feature distributions with temperature &tau; = 2.0 to eliminate outlier noise:<br>'
        '<div style="background:#FAF5FF; padding:8px; border-radius:6px; font-family:monospace; font-size:12px; margin:4px 0;">'
        'p<sub>S</sub> = Softmax(F<sub>S</sub> / &tau;), &nbsp; p<sub>T</sub> = Softmax(F<sub>T</sub> / &tau;), &nbsp; p<sub>M</sub> = 0.5 &times; (p<sub>S</sub> + p<sub>T</sub>)<br>'
        'L<sub>JSD</sub> = 0.5 &times; &tau;<sup>2</sup> &times; [ D<sub>KL</sub>(p<sub>S</sub> || p<sub>M</sub>) + D<sub>KL</sub>(p<sub>T</sub> || p<sub>M</sub>) ]'
        '</div>'
        'JSD is bounded in [0, 1], symmetric, and provably maximizes Mutual Information I(F<sub>S</sub>; F<sub>T</sub>).'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#7C3AED;strokeWidth=2;fontSize=11;',
        parent=p3_p2
    )

    p3_total = p3.add_box(
        1220, 770, 1070, 160,
        '<b>3. Total Student Distillation Objective</b><br>'
        '<div style="background:#ECFDF5; padding:10px; border-radius:6px; font-family:monospace; font-size:12px; margin:6px 0; color:#065F46;">'
        'L<sub>distill</sub> = &varpi;(L<sub>iou</sub>) &times; [ || t<sub>S</sub> - t<sub>T</sub> ||<sub>2</sub><sup>2</sup> + &lambda;<sub>mi</sub> &times; L<sub>JSD</sub>(F<sub>S</sub>, F<sub>T</sub>) ]<br>'
        '<b>L<sub>total</sub><sup>P2</sup> = L<sub>det</sub>(Student) + w<sub>distill</sub> &times; L<sub>distill</sub></b> &nbsp;&nbsp; (w<sub>distill</sub> = 0.0001, &lambda;<sub>mi</sub> = 0.5)'
        '</div>'
        '<span style="font-size:10px; color:#065F46; font-weight:bold;">&check; Zero inference overhead: Teacher network and distillation losses are 100% discarded after training!</span>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=2.5;fontSize=11;',
        parent=p3_p2
    )

    # Page 3 Explicit Dataflow Edges
    p3.add_edge(p3_masks, p3_pass, 'Forward Clean & Masked (Z, X)')
    p3.add_edge(p3_pass, p3_loss1, 'Compute Invariance')
    p3.add_edge(p3_loss1, p3_det, '+ Multi-task Det Loss')
    p3.add_edge(p3_models, p3_adapt, 'GIoU & EMA Stats')
    p3.add_edge(p3_adapt, p3_jsd, 'Weighted Features')
    p3.add_edge(p3_jsd, p3_total, 'Total Loss Backprop')
    p3.add_edge(
        p3_det, p3_models,
        'Phase 1 Weights &rarr; Frozen Teacher in Phase 2',
        'edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#7C3AED;strokeWidth=2;dashed=1;fontColor=#5B21B6;'
    )

    # =========================================================================
    # PAGE 4: Edge Deployment & TensorRT Execution Graph
    # =========================================================================
    p4 = DiagramBuilder('p4_deploy', '4. Edge Deployment & TensorRT', width=2360, height=1320)

    p4.add_box(
        40, 30, 2280, 95,
        '<div style="font-size: 22px; font-weight: bold; color: #0F172A; margin-bottom: 4px;">'
        'Edge Deployment &amp; NVIDIA TensorRT Static Graph Optimization'
        '</div>'
        '<div style="font-size: 13px; color: #334155;">'
        'Transformation from PyTorch Training Topology to a Fully-Fused Single-Stream Static TensorRT Engine for Real-Time UAV Tracking'
        '</div>',
        'rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;shadow=1;'
    )

    # Step 1: Export Pipeline
    c1_deploy = p4.add_box(
        40, 145, 680, 1130,
        '<b>1. DEPLOYMENT PIPELINE</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=14;fontColor=#1E40AF;rounded=1;shadow=1;'
    )
    e1 = p4.add_box(70, 200, 620, 80,
        '<b>Step 1: Checkpoint Loading &amp; Structural Fusion</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; Load student weights: <code>model.load_state_dict(...)</code><br>'
        '&bull; Invoke: <code>model.switch_to_deploy()</code> &rarr; Fuses 3 Conv/BN branches into 1 DWConv(5&times;5)<br>'
        '&bull; Set: <code>model.eval()</code>'
        '</span>',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.5;fontSize=11;',
        parent=c1_deploy
    )
    e2 = p4.add_box(70, 300, 620, 80,
        '<b>Step 2: ONNX Static Graph Export</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; <code>torch.onnx.export(model, (z, x), "birtrack.onnx", opset_version=17)</code><br>'
        '&bull; Explicit static shapes: z=[1, 3, 128, 128], x=[1, 3, 256, 256]<br>'
        '&bull; Zero dynamic axes &bull; Zero conditional control flow nodes'
        '</span>',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.5;fontSize=11;',
        parent=c1_deploy
    )
    e3 = p4.add_box(70, 400, 620, 95,
        '<b>Step 3: NVIDIA TensorRT Engine Build</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; <code>trtexec --onnx=birtrack.onnx --saveEngine=birtrack.engine --fp16</code><br>'
        '&bull; Layer fusion: Combines Conv + Bias + SiLU into 1 CUDA kernel<br>'
        '&bull; Constant folding &amp; dead-code elimination<br>'
        '&bull; INT8 Calibration (optional, PTQ with calibration cache)'
        '</span>',
        'rounded=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.5;fontSize=11;',
        parent=c1_deploy
    )
    e4 = p4.add_box(70, 515, 620, 80,
        '<b>Step 4: Real-Time UAV Edge Runtime</b><br>'
        '<span style="font-size:10px; color:#475569;">'
        '&bull; Python / C++ TensorRT Execution Context<br>'
        '&bull; Template Z processed once on Frame 0, cached in GPU VRAM<br>'
        '&bull; Sequential search updates: <b>&lt; 3.5ms per frame (285+ FPS)</b>'
        '</span>',
        'rounded=1;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=11;fontColor=#065F46;',
        parent=c1_deploy
    )

    p4.add_edge(e1, e2)
    p4.add_edge(e2, e3)
    p4.add_edge(e3, e4)

    # Step 2: Kernel Fusion Diagram
    c2_fusion = p4.add_box(
        760, 145, 780, 1130,
        '<b>2. TENSORRT KERNEL FUSION IN ACTION</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=2;fontSize=14;fontColor=#B45309;rounded=1;shadow=1;'
    )
    box_unfused = p4.add_box(
        790, 200, 720, 160,
        '<b>Before Optimization: Unfused Naive Graph</b><br>'
        '<table style="width:100%; border-collapse:collapse; font-size:10px; text-align:center; margin-top:4px;" border="1">'
        '<tr style="background:#FEE2E2;"><td>Kernel 1: DWConv 5&times;5</td><td>&rarr; VRAM Write &rarr; VRAM Read</td><td>Kernel 2: BatchNorm 1</td></tr>'
        '<tr style="background:#FEE2E2;"><td>Kernel 3: DWConv 3&times;3</td><td>&rarr; VRAM Write &rarr; VRAM Read</td><td>Kernel 4: BatchNorm 2</td></tr>'
        '<tr style="background:#FEE2E2;"><td>Kernel 5: Identity BN</td><td>&rarr; VRAM Write &rarr; VRAM Read</td><td>Kernel 6: Add</td></tr>'
        '<tr style="background:#FEE2E2;"><td>Kernel 7: SiLU Activation</td><td>&rarr; VRAM Write &rarr; VRAM Read</td><td>Kernel 8: Pointwise Conv</td></tr>'
        '</table>'
        '<div style="text-align:left; font-size:10px; color:#B91C1C; margin-top:6px;">'
        '&times; 8 separate GPU kernel launches &bull; Excessive DRAM read/write traffic &bull; High latency jitter'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#EF4444;strokeWidth=1.5;fontSize=11;',
        parent=c2_fusion
    )

    box_fused = p4.add_box(
        790, 390, 720, 180,
        '<b>After BiRTrack Rep-BLGR &amp; TensorRT Optimization: Fully Fused Node</b><br>'
        '<div style="background:#ECFDF5; border:2px solid #10B981; border-radius:6px; padding:12px; margin-top:6px;">'
        '<div style="font-size:12px; font-weight:bold; color:#065F46;">Fused CUDA Kernel: trt_fused_dwconv5x5_bias_silu</div>'
        '<span style="font-size:10px; color:#047857;">'
        '&bull; Single input load from SRAM &rarr; Register-level MAC computation &rarr; In-register SiLU activation<br>'
        '&bull; <b>1 single GPU kernel launch</b> replaces 8 separate kernels!<br>'
        '&bull; Eliminates 87.5% of kernel launch overhead and 70% of DRAM memory bandwidth pressure.'
        '</span>'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=2;fontSize=11;',
        parent=c2_fusion
    )
    p4.add_edge(box_unfused, box_fused, 'switch_to_deploy() + TensorRT Layer Fusion')

    p4.add_box(
        790, 600, 720, 240,
        '<b>Memory Hierarchy Impact on Edge Devices (Jetson Orin Nano / Xavier NX)</b><br>'
        '<table style="width:100%; border-collapse:collapse; font-size:10px; text-align:center; margin-top:6px;" border="1">'
        '<tr style="background:#F1F5F9; font-weight:bold;"><td>Approach</td><td>Memory Access Pattern</td><td>DRAM Bandwidth Used</td><td>Edge FPS (Orin Nano)</td><td>Status</td></tr>'
        '<tr><td><b>Dynamic Pruning (Aba-ViTrack)</b></td><td>Unstructured, variable token count</td><td>34.2 GB/s</td><td>~35 FPS</td><td><span style="color:#B91C1C;">Memory Bound</span></td></tr>'
        '<tr><td><b>Dynamic Layers (Control Flow)</b></td><td>Branch divergence, broken graph</td><td>31.8 GB/s</td><td>~38 FPS</td><td><span style="color:#B91C1C;">Fusion Blocked</span></td></tr>'
        '<tr><td><b>ORTrack-D (Cut ViT)</b></td><td>Static 100%, 8 layers</td><td>21.4 GB/s</td><td>~78 FPS</td><td><span style="color:#D97706;">Accuracy Low (-4.0%)</span></td></tr>'
        '<tr style="background:#D1FAE5; font-weight:bold; color:#065F46;"><td><b>BiRTrack (Ours)</b></td><td>Static 100% + Fused Rep-BLGR</td><td>21.6 GB/s</td><td><b>~75+ FPS</b></td><td><span style="color:#059669;">Optimal Pareto Frontier</span></td></tr>'
        '</table>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;strokeWidth=1.5;fontSize=11;',
        parent=c2_fusion
    )

    # Step 3: Experimental Checklist
    c3_check = p4.add_box(
        1580, 145, 740, 1130,
        '<b>3. DEPLOYMENT &amp; VERIFICATION CHECKLIST</b>',
        'swimlane;fontStyle=1;align=center;verticalAlign=top;childLayout=stackLayout;horizontal=1;startSize=34;horizontalStack=0;resizeParent=1;resizeParentMax=0;resizeLast=0;collapsible=0;marginBottom=0;html=1;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=14;fontColor=#065F46;rounded=1;shadow=1;'
    )
    p4.add_box(
        1610, 200, 680, 480,
        '<b>Pre-Flight Verification Protocol:</b><br>'
        '<div style="text-align:left; font-size:11px; line-height:1.7; color:#1E293B; margin-top:8px;">'
        '&check; <b>1. Numerical Equivalence Check (tools/verify_reparam.py):</b><br>'
        '<span style="font-size:10px; color:#475569; margin-left:16px;">Run verify script to guarantee max absolute error between train and deploy &lt; 1e-4.</span><br>'
        '&check; <b>2. Weight Checksum Integrity:</b><br>'
        '<span style="font-size:10px; color:#475569; margin-left:16px;">Verify MD5 hash of student weights matches training output checkpoint.</span><br>'
        '&check; <b>3. TensorRT FP16 Latency Benchmark:</b><br>'
        '<span style="font-size:10px; color:#475569; margin-left:16px;">Validate average latency &le; 3.5ms on RTX GPU and &le; 13.3ms on Jetson Orin Nano (&gt; 75 FPS).</span><br>'
        '&check; <b>4. Zero Memory Leak over Long Sequences:</b><br>'
        '<span style="font-size:10px; color:#475569; margin-left:16px;">Run 2000-frame video sequence to verify GPU VRAM allocation remains flat.</span><br>'
        '&check; <b>5. Accuracy Benchmark Evaluation:</b><br>'
        '<span style="font-size:10px; color:#475569; margin-left:16px;">Evaluate on standard UAV benchmarks: UAV123 (68.0% Succ), VisDrone2018 (88.2% Prec), UAVDT.</span>'
        '</div>',
        'rounded=1;fillColor=#FFFFFF;strokeColor=#10B981;strokeWidth=2;fontSize=11;',
        parent=c3_check
    )

    # Save to XML string and write to file
    raw_xml = ET.tostring(mxfile, encoding='utf-8')
    dom = minidom.parseString(raw_xml)
    pretty_xml = dom.toprettyxml(indent='  ')

    target_path = os.path.abspath('concepts/Birtrack_Architecture.drawio')
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

    print(f"Successfully generated BiRTrack Draw.io diagram at: {target_path}")
    print(f"File size: {os.path.getsize(target_path)} bytes")

if __name__ == '__main__':
    create_drawio_diagram()
