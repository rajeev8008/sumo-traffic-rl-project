#!/usr/bin/env python
"""Quick script to check mg_road network details"""

import gzip
import xml.etree.ElementTree as ET

# Read compressed network file
with gzip.open('SUMO_Trinity_Traffic_sim/osm.net.xml.gz', 'rt') as f:
    tree = ET.parse(f)

root = tree.getroot()

# Find all traffic lights
print("=" * 70)
print("TRAFFIC LIGHTS IN NETWORK:")
print("=" * 70)
for tl in root.findall('.//tlLogic'):
    tl_id = tl.get('id')
    print(f"  ID: {tl_id}")
    
    # Find which junction this TL controls
    for connection in root.findall('.//connection'):
        tl_ref = connection.get('tl')
        if tl_ref == tl_id:
            from_edge = connection.get('from')
            to_edge = connection.get('to')
            print(f"    From: {from_edge} -> To: {to_edge}")
            break

# Find all edges and their lanes
print("\n" + "=" * 70)
print("EDGES AND LANES:")
print("=" * 70)
for edge in root.findall('.//edge'):
    edge_id = edge.get('id')
    lanes = edge.findall('lane')
    if lanes:
        print(f"\nEdge: {edge_id}")
        for lane in lanes:
            lane_id = lane.get('id')
            print(f"  Lane: {lane_id}")
