"""Compact manuscript schematic: a shared semantic source and two contrasts.

All coordinates are conceptual construction, not experimental embeddings.
Python / Matplotlib only; Times New Roman is required for matching typography.
"""
from pathlib import Path
import argparse,json,os,re
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--qa-dir',type=Path,default=ROOT/'qa')
QA=parser.parse_args().qa_dir
QA.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(QA/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from PIL import Image
from audit_panel_alignment import (require_matplotlib_panel_alignment,
    matplotlib_layout_manifest,audit_layout_manifest,write_json_report,exit_code)

W,H=160,64
HEADER_Y=57.4
INK='#2D3033'; MUTED='#697176'
REF='#718464'; REF_CONTEXT='#95A58B'
BLUE='#4B6F9B'; BLUE_CONTEXT='#7894B5'; DRIFT=MUTED
PT=25.4/72
TARGET_SHIFT=4.8
SYMBOLS=json.loads((Path(__file__).with_name('math_symbols.json')).read_text(encoding='utf-8'))
plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman'],'font.size':8.5,
    'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
    'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
    'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman',
    'mathtext.cal':'Times New Roman','mathtext.fallback':None,
    'pdf.fonttype':42,'ps.fonttype':3,'svg.fonttype':'none',
    'savefig.facecolor':'white','text.color':INK,'axes.linewidth':.5,
    'lines.linewidth':.8,
})

def arrow(ax,start,end,color,width=.85,head=4.0,dashed=False,
          radii=(0,0),gap=.45,gid=None):
    """Independent solid arrowhead, fitted dashes, and physical node clearances.

    Coordinates and clearances are millimetres; widths and heads are points.
    Splitting shaft and head prevents dash styles from fragmenting the triangle.
    """
    a,b=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    direction=(b-a)/np.linalg.norm(b-a)
    normal=np.array([-direction[1],direction[0]])
    a=a+direction*(radii[0]+gap)
    tip=b-direction*(radii[1]+gap)
    base=tip-direction*head*PT
    triangle=np.array([tip,base+normal*head*.38*PT,base-normal*head*.38*PT])
    # A tiny underlap into the filled head prevents a renderer hairline seam.
    shaft_end=base+direction*.12*PT
    length=np.linalg.norm(shaft_end-a)
    assert length>0
    if dashed:
        dash,gap_length=2.8*PT,2.0*PT
        n=max(2,round((length+gap_length)/(dash+gap_length)))
        actual_dash=(length-(n-1)*gap_length)/n
        for k in range(n):
            lo=k*(actual_dash+gap_length)
            ends=np.array([a+direction*lo,a+direction*(lo+actual_dash)])
            line=ax.plot(ends[:,0],ends[:,1],color=color,lw=width,
                         solid_capstyle='butt',zorder=3)[0]
            if gid:line.set_gid(f'{gid}-dash-{k}')
    else:
        line=ax.plot([a[0],shaft_end[0]],[a[1],shaft_end[1]],color=color,
                     lw=width,solid_capstyle='butt',zorder=3)[0]
        if gid:line.set_gid(gid+'-shaft')
    p=Polygon(triangle,closed=True,facecolor=color,edgecolor='none',zorder=4)
    if gid:p.set_gid(gid+'-head')
    ax.add_patch(p)
    ax._schematic['edges'].append({'id':gid,'start':a.tolist(),'end':tip.tolist(),
        'head':triangle.tolist(),'width_mm':width*PT,
        'start_node':np.asarray(start).tolist(),'end_node':np.asarray(end).tolist(),
        'node_gap_mm':gap,'dashed_shaft':dashed,'solid_head':True})
    return p

def source_flow(fig,x1,x2):
    base=x2-3.7*PT
    fig.add_artist(Line2D([x1/W,(base+.12*PT)/W],[HEADER_Y/H]*2,
        transform=fig.transFigure,lw=.7,color=MUTED,solid_capstyle='butt'))
    fig.add_artist(Polygon([[x2/W,HEADER_Y/H],[base/W,(HEADER_Y+1.3*PT)/H],
        [base/W,(HEADER_Y-1.3*PT)/H]],transform=fig.transFigure,
        facecolor=MUTED,edgecolor='none',closed=True))

def node_radius(central,reference):
    diameter=(5.7 if central else 4.6) if reference else (5.6 if central else 4.6)
    stroke=.9 if reference else .5
    return (diameter/2+stroke/2)*PT

def graph(ax,points,reference,tag):
    for k,(i,j) in enumerate([(0,1),(1,2),(2,3)]):
        color=(REF if k==1 else REF_CONTEXT) if reference else (BLUE if k==1 else BLUE_CONTEXT)
        arrow(ax,points[i],points[j],color,
              width=(.95 if k==1 else .65) if reference else (1.25 if k==1 else .85),
              head=4.0 if k==1 else 3.5,dashed=reference,
              radii=(node_radius(i in [1,2],reference),node_radius(j in [1,2],reference)),
              gid=f'{tag}-local-contrast-{k}')
    for k,p in enumerate(points):
        central=k in [1,2]
        color=REF if reference else BLUE
        m=ax.plot(*p,'o',ms=(5.7 if central else 4.6) if reference else (5.6 if central else 4.6),
            mfc='white' if reference else color,mec=color if reference else 'white',
            mew=.9 if reference else .5,zorder=7 if reference else 8)[0]
        m.set_gid(f'{tag}-class-{k}')
        ax._schematic['nodes'].append({'id':m.get_gid(),'xy':p.tolist(),
                                      'radius_mm':node_radius(central,reference)})

def box_distance(point,bounds):
    x,y=point; x0,y0,x1,y1=bounds
    return np.hypot(max(x0-x,0,x-x1),max(y0-y,0,y-y1))

def label(ax,node,key,reference,direction,gap=1.0):
    """Mother-diagram Times italic outlines, with its explicit script metrics."""
    parts=[]
    for char,size,offset in [(key[0],SYMBOLS['base_size_pt'],(0,0)),
              (key[1],SYMBOLS['subscript_size_pt'],
               (SYMBOLS['subscript_x_pt'][key],SYMBOLS['subscript_baseline_pt']))]:
        glyph=SYMBOLS['glyphs'][char]
        path=MplPath(glyph['vertices'],glyph['codes'])
        transform=Affine2D().scale(size/1000*PT).translate(*(np.array(offset)*PT))
        parts.append(transform.transform_path(path))
    combined=MplPath.make_compound_path(*parts)
    bounds=combined.get_extents()
    half=np.array([bounds.width,bounds.height])/2
    node=np.asarray(node,dtype=float)
    direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
    def rect(distance):
        center=node+direction*distance
        return np.r_[center-half,center+half]
    low,high=0.,10.
    for _ in range(50):
        middle=(low+high)/2
        if box_distance(node,rect(middle))<node_radius(True,reference)+gap:low=middle
        else:high=middle
    center=node+direction*high
    transform=Affine2D().translate(*(center-np.array([bounds.x0+bounds.width/2,
                                                     bounds.y0+bounds.height/2])))
    path=transform.transform_path(combined)
    p=PathPatch(path,facecolor=REF if reference else BLUE,edgecolor='none',zorder=10)
    p.set_gid(f'{ax.get_gid()}-math-{key}')
    ax.add_patch(p)
    ax._schematic['labels'].append({'id':p.get_gid(),'key':key,'node':node.tolist(),
        'bounds_mm':rect(high).tolist(),'node_gap_mm':gap,
        'base_font':SYMBOLS['font'],'subscript_font':SYMBOLS['font'],
        'base_size_pt':SYMBOLS['base_size_pt'],'subscript_size_pt':SYMBOLS['subscript_size_pt'],
        'subscript_dx_pt':SYMBOLS['subscript_x_pt'][key],
        'subscript_baseline_pt':SYMBOLS['subscript_baseline_pt']})
    return p

def normalize_svg(path):
    s=path.read_text(encoding='utf-8')
    s=re.sub(r'<!DOCTYPE\s+svg\s+PUBLIC.*?>\s*','',s,flags=re.S)
    s=re.sub(r'\s+transform="rotate\(-?0(?:\.0+)? [^"\n]+\)"','',s)
    def fix(m):
        style=m.group(2);size=re.search(r'font-size:\s*([0-9.]+)px\s*;?',style)
        if not size:return m.group(0)
        style=style[:size.start()]+style[size.end():]
        return m.group(1)+'font-size="'+size.group(1)+'" style="'+style.strip()+'"'+m.group(3)
    s=re.sub(r'(<(?:text|tspan)\b[^>]*?)style="([^"]*)"([^>]*>)',fix,s)
    path.write_text(s,encoding='utf-8')

def main():
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    source_labels=[]
    for x,lines in [(27,'LLM-assisted\nhierarchical descriptions'),
                    (80,'Frozen CLIP\ntext encoder'),
                    (133,'Fixed semantic\nreferences')]:
        source_labels.append(fig.text(x/W,HEADER_Y/H,lines,fontsize=8.5,ha='center',va='center',
                 linespacing=1.3,color=INK))
    fig.canvas.draw()
    source_boxes=[t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) for t in source_labels]
    for left,right in zip(source_boxes[:-1],source_boxes[1:]):
        source_flow(fig,left.x1*25.4+2.4,right.x0*25.4-2.4)
    handles=[Line2D([],[],lw=.85,ls=(0,(3.3,2.3)),color=REF,marker='o',ms=4.7,
                    mfc='white',mec=REF,label='CLIP reference'),
             Line2D([],[],lw=1.1,color=BLUE,marker='o',ms=4.7,
                    mfc=BLUE,mec='white',mew=.45,label='Prompted text feature')]
    fig.legend(handles=handles,ncol=2,loc='center',bbox_to_anchor=(.5,47.6/H),
               frameon=False,fontsize=8.3,handlelength=2.0,handletextpad=.6,
               columnspacing=2.0,labelcolor=INK)
    axes=[]
    for x in [4,84]:
        ax=fig.add_axes([x/W,4/H,72/W,40/H])
        ax.set(xlim=(0,72),ylim=(0,40),aspect='equal');ax.set_axis_off();axes.append(ax)
        ax._schematic={'edges':[],'nodes':[],'labels':[]}
    q=np.array([[16.,12.],[24.,32.],[50.,29.],[58.,10.]])
    risk=np.array([[21.,17.],[30.,25.],[45.,20.],[55.,7.]])
    q[:,0]=36+(q[:,0]-36)*1.3
    risk[:,0]=36+(risk[:,0]-36)*1.3
    target=q+np.array([0.,TARGET_SHIFT])
    for panel,ax,t in zip('ab',axes,[risk,target]):
        ax.set_gid(panel)
        graph(ax,q,True,panel+'-reference')
        graph(ax,t,False,panel+'-prompted')
    arrow(axes[0],q[1],risk[1],DRIFT,width=.7,head=3.5,
          radii=(node_radius(True,True),node_radius(True,False)),gid='a-anchor-displacement')
    for node,key,ref,direction in [(q[1],'qj',True,[-1,.6]),(q[2],'qi',True,[1,.6]),
                        (risk[1],'tj',False,[.2,-1]),(risk[2],'ti',False,[-.1,-1])]:
        label(axes[0],node,key,ref,direction)
    for node,key,ref,direction in [(q[1],'qj',True,[.65,-1]),(q[2],'qi',True,[-.5,-1]),
                        (target[1],'tj',False,[-1,.35]),(target[2],'ti',False,[1,.35])]:
        label(axes[1],node,key,ref,direction)
    captions=[]
    for ax,title in zip(axes,['(a) Potential semantic drift','(b) SPCP preservation target']):
        captions.append(ax.text(36,0,title,ha='center',va='center',fontsize=9,
                                 weight='bold',clip_on=False,color=INK))
    fig.canvas.draw()
    (QA/'element_geometry.json').write_text(json.dumps(
        {ax.get_gid():ax._schematic for ax in axes},indent=2),encoding='utf-8')
    # Measure the parenthesized bottom captions explicitly at the same tolerance.
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['a','b'],row_groups=[['a','b']],
        json_out=QA/'plot_area_alignment.json',tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True)
    manifest=matplotlib_layout_manifest(fig,axes=axes,panel_ids=['a','b'],row_groups=[['a','b']])
    for panel,artist in zip(manifest['panels'],captions):
        display=artist.get_transform().transform(artist.get_position())
        inches=fig.dpi_scale_trans.inverted().transform(display)
        panel['panel_label']=artist.get_text()
        panel['panel_label_anchor_pt']=(inches*72).tolist()
    report=audit_layout_manifest(manifest,tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True)
    assert exit_code(report,strict=True)==0,report
    write_json_report(report,QA/'alignment.json')
    stem=ROOT/'SPCP_introduction_motivation'
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    with plt.rc_context({'svg.fonttype':'path'}):fig.savefig(str(stem)+'.outlined.svg')
    fig.savefig(str(stem)+'.eps')
    fig.savefig(str(stem)+'.png',dpi=600)
    fig.savefig(QA/'preview.png',dpi=300)
    for suffix in ['.svg','.outlined.svg']:normalize_svg(Path(str(stem)+suffix))
    with Image.open(str(stem)+'.png') as im:im.convert('L').save(QA/'greyscale.png')
    # The 2D sketch admits a unit-vector realization in four dimensions.
    # These values verify construction; they are not experimental data.
    xy=(q-np.array([37.,21.]))*.006
    D=np.array([0.,TARGET_SHIFT*.006,TARGET_SHIFT*.006,0.])
    z=(-np.dot(D,D)/2-xy@D[:2])/D[2]
    refs=np.c_[xy,z,np.sqrt(1-(xy*xy).sum(1)-z*z)]
    kept=refs+D
    rxy=(risk-np.array([37.,21.]))*.006
    moved=np.c_[rxy,z,np.sqrt(1-(rxy*rxy).sum(1)-z*z)]
    assert all(np.allclose(np.linalg.norm(v,axis=1),1) for v in [refs,kept,moved])
    assert np.allclose(np.diff(refs,axis=0),np.diff(kept,axis=0))
    assert not np.allclose(np.diff(refs,axis=0),np.diff(moved,axis=0))
    data={'not_experimental_data':True,'reference_points':q.tolist(),
          'adaptation_risk_points':risk.tolist(),'preservation_target_points':target.tolist(),
          'central_edge':'j to i','full_difference_vectors_preserved_in_target':True,
          'unit_vector_realization_checked':True,
          'scope':'Intended semantic preservation target, not a fitted embedding or a guaranteed recognition outcome.'}
    (QA/'construction.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    plt.close(fig)
    print('Exported 160 x 64 mm manuscript figure with two bottom-captioned panels.')

if __name__=='__main__':main()

