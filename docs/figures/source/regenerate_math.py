"""Regenerate SVG math groups from their editable NewTX LaTeX manifest.

Only math glyphs are replaced. Diagram geometry, labels and connectors remain
in ../*.svg; individual mathematics is editable in math_labels.json.
"""
from pathlib import Path
import argparse, copy, json, re, xml.etree.ElementTree as ET
from math_renderer import render_math

ROOT=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg'; XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS); ET.register_namespace('xlink',XL)

def update(output, tectonic, class_file=None, only_cached=False):
    items=json.loads((ROOT/'math_labels.json').read_text())
    metrics=render_math(items,output,tectonic,class_file,only_cached)
    trees={name:ET.parse(ROOT.parent/name) for name in {v['figure'] for v in items}}
    for item in items:
        key=item['id']; tree=trees[item['figure']]
        group=next(e for e in tree.iter() if e.get('data-math-key')==key)
        group.set('data-latex',item['latex'])
        for c in list(group): group.remove(c)
        m=metrics[key]; svg=ET.parse(Path(output)/(key+'.svg')).getroot()
        # Prefix reusable TeX glyph IDs so labels cannot shadow each other.
        for e in svg.iter():
            if 'id' in e.attrib:e.set('id',key+'-'+e.get('id'))
            for attr,val in list(e.attrib.items()):
                if attr.endswith('href') and val.startswith('#'):e.set(attr,'#'+key+'-'+val[1:])
                if 'url(#' in val:e.set(attr,val.replace('url(#','url(#'+key+'-'))
        # Remove the white baseline marker used only during measurement.
        for p in svg.iter():
            for e in list(p):
                if e.get('fill') in ['#ffffff','#FFFFFF','white']:p.remove(e)
        x=item.get('x',0);y=item.get('y',0)
        if item.get('anchor')=='middle':x-=m['width']/2
        inner=ET.SubElement(group,'{'+NS+'}g',{'transform':f'translate({x:.6f} {y-m["baseline"]:.6f})','fill':'#252525'})
        for child in svg:inner.append(copy.deepcopy(child))
    for name,tree in trees.items():tree.write(ROOT.parent/name,encoding='utf-8',xml_declaration=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tectonic',default='tectonic')
    p.add_argument('--class-file',type=Path)
    p.add_argument('--only-cached',action='store_true')
    a=p.parse_args();update(a.output,a.tectonic,a.class_file,a.only_cached)
