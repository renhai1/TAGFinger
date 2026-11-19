#!/usr/bin/env python3
"""Original motion-design sample: continuous graph, camera moves, depth and type.

No scientific values are invented: the probability responses and text edit are
explicitly schematic. Procedural illustrations, music bed and sound accents
are original. Narration is synthesized locally. No external visual asset is used.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
import re
import subprocess
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT=Path(__file__).resolve().parent
STORY=json.loads((ROOT/'storyboard.json').read_text())
OUT=ROOT.parents[1]/'images'
W,H=1920,1080
PAPER='#F0F1F6'
NIGHT='#111322'
INK='#172032'
GREY='#7E8699'
WHITE='#F9FAFF'
PURPLE='#6856E8'
MINT='#27B79A'
LIME='#CCF481'
VIOLET='#BCB4FF'
MONO='/System/Library/Fonts/Menlo.ttc'
SANS='/System/Library/Fonts/HelveticaNeue.ttc'
DISPLAY='/System/Library/Fonts/Avenir Next.ttc'

# A graph that persists across shots; a chosen region is never a new random graph.
WORLD=np.array([
 [-320,-95,0],[-200,-192,12],[-52,-216,-12],[110,-181,5],[288,-135,-7],
 [-374,35,-15],[-229,-20,0],[-63,-54,8],[112,-48,-10],[319,7,5],
 [-315,182,-8],[-151,122,3],[20,93,0],[178,134,9],[354,149,-4],
 [-89,278,-7],[80,273,3],[257,280,4],[-406,216,-5],[405,-145,10]
],dtype=float)
LINKS=[(0,1),(0,5),(0,6),(1,2),(1,6),(2,3),(2,7),(3,4),(3,8),(4,9),(4,19),
 (5,6),(5,10),(6,7),(6,11),(7,8),(7,11),(7,12),(8,9),(8,12),(8,13),
 (9,14),(10,11),(10,18),(11,12),(11,15),(12,13),(12,16),(13,14),
 (13,16),(13,17),(14,17),(15,16),(16,17)]
SELECTED={7,11,12}
TARGET=7


def sat(x):return max(0.,min(1.,float(x)))
def ease(x):x=sat(x);return x*x*(3-2*x)
def out(x):return 1-(1-sat(x))**4
def smooth(a,b,t):return a+(b-a)*ease(t)
def color(hexcode):
    if isinstance(hexcode,tuple):return hexcode[:3]
    return tuple(int(hexcode[i:i+2],16) for i in (1,3,5))
def mix(a,b,p):return tuple(round(x+(y-x)*sat(p)) for x,y in zip(color(a),color(b)))
def lerp(a,b,p):return tuple(x+(y-x)*p for x,y in zip(a,b))


@functools.lru_cache(maxsize=512)
def font(size,weight='regular'):
    if weight=='mono':return ImageFont.truetype(MONO,size,index=0)
    if weight=='display':return ImageFont.truetype(DISPLAY,size,index=0)
    if weight=='thin':return ImageFont.truetype(SANS,size,index=7)
    if weight=='bold':return ImageFont.truetype(SANS,size,index=1)
    return ImageFont.truetype(SANS,size,index=0)


@functools.lru_cache(maxsize=512)
def text_sprite(text,size,weight,rgb):
    f=font(size,weight)
    box=f.getbbox(text)
    im=Image.new('RGBA',(max(1,box[2]-box[0]+8),max(1,box[3]-box[1]+8)))
    ImageDraw.Draw(im).text((4-box[0],4-box[1]),text,font=f,fill=rgb+(255,))
    return im


@functools.lru_cache(maxsize=16)
def background(width,dark):
    hh=round(width*9/16)
    yy,xx=np.mgrid[0:hh,0:width].astype(np.float32)
    xx/=width;yy/=hh
    base=np.array(color(NIGHT if dark else PAPER),dtype=np.float32)
    glow1=np.exp(-(((xx-.8)/.42)**2+((yy-.33)/.45)**2)*2.5)
    glow2=np.exp(-(((xx-.36)/.50)**2+((yy-.92)/.5)**2)*2.5)
    if dark:
        change=glow1[:,:,None]*np.array([18,9,40])+glow2[:,:,None]*np.array([0,12,11])
    else:
        change=glow1[:,:,None]*np.array([-8,-12,-2])+glow2[:,:,None]*np.array([3,3,4])
    rng=np.random.default_rng(47)
    noise=rng.normal(0,.24,(hh,width,1))
    img=np.clip(base+change+noise,0,255).astype(np.uint8)
    return Image.fromarray(img,'RGB')


@functools.lru_cache(maxsize=128)
def shadow_sprite(w,h,blur,opacity):
    pad=blur*3
    mask=Image.new('L',(w+pad*2,h+pad*2))
    ImageDraw.Draw(mask).rounded_rectangle((pad,pad,pad+w,pad+h),radius=24,fill=opacity)
    mask=mask.filter(ImageFilter.GaussianBlur(blur))
    im=Image.new('RGBA',mask.size,(13,22,43,0));im.putalpha(mask)
    return im


@functools.lru_cache(maxsize=64)
def sphere_sprite(diameter,rgb,dark=False):
    pad=round(diameter*.35);size=diameter+2*pad;r=diameter/2
    yy,xx=np.mgrid[0:size,0:size].astype(np.float32)
    xx=(xx-(size-1)/2)/r;yy=(yy-(size-1)/2)/r
    rr=xx*xx+yy*yy;z=np.sqrt(np.maximum(0,1-rr))
    lam=np.maximum(0,-.47*xx-.63*yy+.62*z)
    spec=np.maximum(0,-.35*xx-.5*yy+.79*z)**28
    base=np.array(rgb,dtype=np.float32)
    shading=.50+.57*lam
    pix=np.clip(base[None,None,:]*shading[:,:,None]+spec[:,:,None]*104,0,255)
    rgba=np.zeros((size,size,4),dtype=np.uint8)
    rgba[:,:,:3]=pix
    rgba[:,:,3]=(np.clip((1-rr)*r,0,1)*255).astype(np.uint8)
    if dark:
        glow=np.exp(-rr*1.55)*70
        glow[rr<1]=0
        rgba[:,:,3]=np.maximum(rgba[:,:,3],glow.astype(np.uint8))
        rgba[rr>=1,:3]=base.astype(np.uint8)
    return Image.fromarray(rgba,'RGBA')


@functools.lru_cache(maxsize=256)
def paper_node(size,selected,dark,number):
    # Small upright text cards with a bevel and a real text hierarchy.
    scale=3
    ww,hh=round(size*1.2)*scale,round(size*1.6)*scale
    im=Image.new('RGBA',(ww+24*scale,hh+24*scale))
    d=ImageDraw.Draw(im)
    x,y=12*scale,12*scale
    fill=(45,43,73,255) if dark else (255,255,255,255)
    edge=(124,111,205,255) if selected else ((73,73,104,255) if dark else (208,214,231,255))
    d.rounded_rectangle((x+3*scale,y+4*scale,x+ww+3*scale,y+hh+4*scale),radius=6*scale,fill=(7,12,34,35 if not dark else 95))
    d.rounded_rectangle((x,y,x+ww,y+hh),radius=6*scale,fill=fill,outline=edge,width=scale)
    accent=color(MINT if selected else PURPLE)
    d.rounded_rectangle((x+7*scale,y+8*scale,x+ww-7*scale,y+11*scale),radius=scale,fill=accent+(255,))
    for j,ratio in enumerate([.88,.73,.90,.54,.80]):
        yy=y+(21+j*7)*scale
        d.rectangle((x+7*scale,yy,x+7*scale+(ww-14*scale)*ratio,yy+1.7*scale),fill=(157,162,185,160))
    # Rotate consistently; motion itself comes from the 3-D camera.
    return im.rotate(-7 if number%3 else 6,resample=Image.Resampling.BICUBIC,expand=True).resize(
        (round(im.width/scale),round(im.height/scale)),Image.Resampling.LANCZOS)


class Stage:
    def __init__(self,width=1920,dark=False):
        self.s=width/1920
        self.im=background(width,dark).convert('RGBA')
        self.d=ImageDraw.Draw(self.im)
        self.dark=dark

    def xy(self,p):return tuple(round(v*self.s) for v in p)
    def line(self,points,fill,width=1):
        self.d.line([self.xy(p) for p in points],fill=fill,
                    width=max(1,round(width*self.s)),joint='curve')
    def rect(self,x,y,w,h,fill,r=0,outline=None,width=1):
        xy=self.xy((x,y,x+w,y+h))
        if r:self.d.rounded_rectangle(xy,radius=round(r*self.s),fill=fill,outline=outline,width=max(1,round(width*self.s)))
        else:self.d.rectangle(xy,fill=fill,outline=outline,width=max(1,round(width*self.s)))
    def ellipse(self,x,y,rx,ry,fill=None,outline=None,width=1):
        self.d.ellipse(self.xy((x-rx,y-ry,x+rx,y+ry)),fill=fill,outline=outline,
                       width=max(1,round(width*self.s)))
    def polygon(self,points,fill):self.d.polygon([self.xy(p) for p in points],fill=fill)
    def paste(self,sprite,x,y,opacity=1,anchor='lt',native=False):
        if opacity<=0:return
        sprite=sprite.copy()
        if self.s!=1 and not native:sprite=sprite.resize((max(1,round(sprite.width*self.s)),max(1,round(sprite.height*self.s))),Image.Resampling.LANCZOS)
        if opacity<.999:sprite.putalpha(sprite.getchannel('A').point(lambda a:round(a*opacity)))
        xx,yy=self.xy((x,y))
        if anchor=='mm':xx-=sprite.width//2;yy-=sprite.height//2
        if anchor=='mt':xx-=sprite.width//2
        if anchor=='rt':xx-=sprite.width
        self.im.alpha_composite(sprite,(xx,yy));self.d=ImageDraw.Draw(self.im)
    def text(self,x,y,text,size=28,fill=INK,weight='regular',opacity=1,anchor='lt',rise=0):
        self.paste(text_sprite(text,round(size*self.s),weight,color(fill)),x,y+rise,opacity,anchor,native=True)
    def tracking(self,x,y,text,size=14,fill=GREY,spacing=3,opacity=1):
        xx=x
        for ch in text:
            self.text(xx,y,ch,size,fill,'mono',opacity)
            xx+=font(size,'mono').getlength(ch)+spacing
    def shadow(self,x,y,w,h,blur=32,opacity=32):
        sprite=shadow_sprite(round(w),round(h),blur,opacity)
        self.paste(sprite,x-blur*3,y-blur*3)
    def sphere(self,x,y,r,fill=PURPLE,opacity=1):
        d=max(6,round(r*2));self.paste(sphere_sprite(d,color(fill),self.dark),x,y,opacity,'mm')
    def bezier(self,points,fill,width=2,progress=1):
        steps=45;p0,p1,p2,p3=points
        curve=[]
        for tt in np.linspace(0,sat(progress),steps):
            it=1-tt
            curve.append((it**3*p0[0]+3*it*it*tt*p1[0]+3*it*tt*tt*p2[0]+tt**3*p3[0],
                          it**3*p0[1]+3*it*it*tt*p1[1]+3*it*tt*tt*p2[1]+tt**3*p3[1]))
        self.line(curve,fill,width)
        return curve
    def marker(self,x,y,label,fill=PURPLE,opacity=1):
        self.rect(x,y+5,5,5,mix(fill,PAPER,1-opacity),2)
        self.tracking(x+17,y,label,12,fill,1.2,opacity)


def ui(stage,chapter,dark=False):
    fg=WHITE if dark else INK
    stage.tracking(98,49,'TAGFINGER',16,fg,2)
    stage.text(1789,48,'SEMANTICS × STRUCTURE',12,'#959AB6' if dark else GREY,'mono',anchor='rt')
    stage.line([(99,85),(1821,85)],'#303045' if dark else '#D5D9E5',1)
    stage.tracking(99,123,chapter,13,VIOLET if dark else PURPLE,2)
    stage.text(99,1007,'ANONYMOUS RESEARCH OVERVIEW',12,'#9298B0' if dark else GREY,'mono')
    stage.text(1815,1007,'METHOD VISUALIZATION / SCHEMATIC',12,'#9298B0' if dark else GREY,'mono',anchor='rt')


def project(world,center=(1270,577),zoom=1.,angle=-.18,tilt=.6,target=(0,0,0)):
    p=world-np.array(target)
    c,s=math.cos(angle),math.sin(angle)
    xx=p[:,0]*c-p[:,1]*s;yy=p[:,0]*s+p[:,1]*c;zz=p[:,2]
    # Camera tilt generates true parallax: cards and nodes move on an inclined plane.
    y=yy*math.cos(tilt)-zz*math.sin(tilt)
    z=yy*math.sin(tilt)+zz*math.cos(tilt)
    perspective=1150/(1150+z)
    return np.column_stack((center[0]+xx*zoom*perspective,
                            center[1]+y*zoom*perspective,
                            perspective,z))


def graph(stage,t,center=(1270,577),zoom=1.,angle=-.18,tilt=.60,
          selection=0.,documents=1.,target=(0,0,0),opacity=1.,flow=0.,floating=0.):
    positions=WORLD.copy()
    for i in range(len(positions)):
        positions[i,2]+=3*math.sin(t*.45+i*.77)
        if i in SELECTED:positions[i,2]+=selection*18
    points=project(positions,center,zoom,angle,tilt,target)
    fg=color(VIOLET if stage.dark else PURPLE)
    edge=mix('#414362' if stage.dark else '#B6BED6',NIGHT if stage.dark else PAPER,1-opacity)
    if selection>0:
        vertices=[]
        for i in (7,12,11):
            xx,yy=points[i,:2];cx,cy=np.mean(points[list(SELECTED),:2],axis=0)
            vertices.append((cx+(xx-cx)*1.75,cy+(yy-cy)*1.75))
        fill=mix('#262445' if stage.dark else '#E1DCF8',NIGHT if stage.dark else PAPER,1-selection*opacity)
        stage.polygon(vertices,fill)
        stage.line(vertices+[vertices[0]],mix(PURPLE,NIGHT if stage.dark else PAPER,1-selection*.45*opacity),1)
    for i,(a,b) in enumerate(LINKS):
        pa,pb=points[a,:2],points[b,:2]
        selected=a in SELECTED and b in SELECTED
        midpoint=(pa+pb)/2;curv=8*math.sin(i*1.11)
        line_color=mix(MINT if selected else edge,edge,1-selection) if selected else edge
        stage.bezier([pa,lerp(pa,midpoint,.8),(midpoint[0],midpoint[1]-curv),pb],line_color,2.7 if selected else 1.4)
        if flow>0 and (i%3==0 or selected):
            f=(t*.17+i*.19)%1
            xx,yy=pa+(pb-pa)*f
            stage.ellipse(xx,yy,2.6,2.6,mix(MINT if selected else PURPLE,NIGHT if stage.dark else PAPER,1-opacity*flow))
    for i in np.argsort(points[:,3])[::-1]:
        x,y,p,z=points[i]
        selected=i in SELECTED
        alpha=opacity*(1 if selected else 1-selection*.55)
        node_color=mix(PURPLE,MINT,selection) if selected else ('#7E71D1' if stage.dark else '#9299C1')
        r=(16 if i==TARGET else 12)*zoom*p
        if selected and selection>.1:
            stage.ellipse(x,y,r+11+2*math.sin(t*1.4),r+7,None,
                          mix(MINT,NIGHT if stage.dark else PAPER,1-selection*.55*alpha),1.2)
        if not stage.dark:
            stage.ellipse(x+5,y+14,r*.95,r*.28,mix('#CBD1E0',PAPER,1-alpha*.55))
        stage.sphere(x,y,r,node_color,alpha)
        if documents>0 and i in (0,2,4,7,10,11,12,14,17):
            sz=max(25,min(78,round(33*zoom*p)))
            sprite=paper_node(sz,selected,stage.dark,int(i))
            stage.paste(sprite,x+12*zoom,y-44*zoom-floating*selection,alpha*documents,'mm')
    return points


def cube(stage,x,y,size=200,opacity=1.,phase=0):
    # Three faces, restrained glass outlines, rim-light and a floating shadow.
    depth=size*.32
    front=[(x-size/2,y-size/2),(x+size/2,y-size/2),(x+size/2,y+size/2),(x-size/2,y+size/2)]
    top=[front[0],(front[0][0]+depth,front[0][1]-depth),
         (front[1][0]+depth,front[1][1]-depth),front[1]]
    side=[front[1],top[2],(front[2][0]+depth,front[2][1]-depth),front[2]]
    background=NIGHT if stage.dark else PAPER
    if not stage.dark:stage.shadow(x-size/2,y+size/2+18,size+depth,30,24,25)
    stage.polygon(top,mix('#424064' if stage.dark else '#C1BFE0',background,1-opacity))
    stage.polygon(side,mix('#24243B' if stage.dark else '#9694BE',background,1-opacity))
    stage.polygon(front,mix('#191D32' if stage.dark else '#252B49',background,1-opacity))
    for poly in (front,top,side):stage.line(poly+[poly[0]],mix('#8D82C7',background,1-opacity*.65),1.3)
    for i in range(3):
        width=size*.54
        by=y-size*.23+i*size*.22
        stage.rect(x-size*.31,by,width,size*.09,mix('#383A5E',background,1-opacity),4)
        stage.rect(x-size*.31,by,width*(.44+.19*math.sin(phase*.3+i)),size*.09,
                   mix('#877BDC',background,1-opacity),4)
    stage.text(x,y+size*.65,'BLACK-BOX GNN',15,WHITE if stage.dark else INK,'mono',opacity,'mt')
    return front


def arrow(stage,x1,y1,x2,y2,fill=PURPLE,progress=1,width=2):
    p=out(progress);xx,yy=lerp((x1,y1),(x2,y2),p)
    stage.line([(x1,y1),(xx,yy)],fill,width)
    angle=math.atan2(y2-y1,x2-x1)
    if p>0:
        stage.line([(xx-10*math.cos(angle-.55),yy-10*math.sin(angle-.55)),(xx,yy),
                    (xx-10*math.cos(angle+.55),yy-10*math.sin(angle+.55))],fill,width)


def opening(stage,t,duration):
    ui(stage,'THE OWNERSHIP QUESTION',True)
    p=out(t/1.4)
    stage.text(94,233,'Did your data',99,WHITE,'thin',p,rise=(1-p)*35)
    p2=out((t-.25)/1.3)
    stage.text(94,356,'train this model?',99,WHITE,'bold',p2,rise=(1-p2)*35)
    line_p=out((t-.8)/1.7)
    stage.rect(99,502,245*line_p,3,LIME,1)
    stage.text(99,560,'The training recipe is hidden.',30,'#ACB0C8','regular',out((t-1.2)/1.3))
    stage.text(99,609,'The model’s behavior is not.',30,'#ACB0C8','regular',out((t-1.5)/1.3))
    points=graph(stage,t,center=(1370,651),zoom=1.0+.02*t,
                  angle=-.27+t*.012,tilt=.66,opacity=out((t-.3)/1.2),flow=out((t-1.4)/2))
    cube(stage,1535,345+5*math.sin(t*.8),180,opacity=out((t-.7)/1.2),phase=t)
    if t>1.3:
        curve=stage.bezier([points[7,:2],(1380,470),(1430,409),(1510,413)],'#79709B',1.7,out((t-1.3)/1.4))
        f=(t*.45)%1
        ii=min(len(curve)-1,round(f*(len(curve)-1)))
        stage.sphere(*curve[ii],4,LIME)
    stage.tracking(98,778,'TEXT-ATTRIBUTE GRAPHS',14,VIOLET,2,opacity=out((t-2)/1.2))
    stage.text(98,824,'Texts on nodes. Relationships on edges.',26,'#B5B8CF',opacity=out((t-2.3)/1.2))


def region(stage,t,duration):
    ui(stage,'01 / STRUCTURAL CONTEXT')
    p=out(t/1.3)
    stage.text(95,228,'Zoom into',104,INK,'thin',p,rise=(1-p)*24)
    stage.text(95,354,'what lasts.',104,INK,'bold',out((t-.2)/1.3),rise=(1-p)*24)
    stage.marker(101,526,'PREDICTIVELY STABLE',MINT, out((t-.7)/1.3))
    stage.text(97,572,'Local text semantics.',31,GREY,opacity=out((t-1.0)/1.2))
    stage.text(97,620,'Neighbor consistency.',31,GREY,opacity=out((t-1.3)/1.2))
    selection=out((t-2.0)/2.5)
    camera=ease((t-2.8)/max(1,duration-4))
    enter=out(t/1.35)
    center=lerp((1370,651),(1350,583),camera*.6+enter*.4)
    target=lerp((0,0,0),(-47,36,0),camera)
    points=graph(stage,t,center,zoom=1.07+.75*camera,
                 angle=-.21+.11*camera,tilt=.66-.20*camera,
                 selection=selection,opacity=1,flow=.6,documents=1,target=target,floating=7)
    if selection>.2:
        cluster=np.mean(points[list(SELECTED),:2],axis=0)
        stage.line([(cluster[0]+55,cluster[1]-55),(1678,341),(1780,341)],mix(MINT,PAPER,1-selection),1.5)
        stage.text(1730,295,'LOCAL REGION',16,MINT,'mono',selection,'mt')
        stage.text(1730,361,'target + neighbors',18,GREY,opacity=selection,anchor='mt')
    # Owner release shown as an immutable small footer, not confused with the copy.
    stage.line([(101,755),(589,755)],'#C4CAD9',1)
    stage.marker(102,786,'RELEASED DATA / UNCHANGED',PURPLE,out((t-.5)/1.4))
    stage.text(100,829,'Edits happen on the audit copy.',25,INK,opacity=out((t-1)/1.4))


def article(stage,x,y,w,h,p=1.,edited=0.,opacity=1.,tilt=0.):
    stage.shadow(x+9,y+22,w,h,30,round(36*opacity))
    stage.rect(x,y,w,h,mix('#FFFFFF',PAPER,1-opacity),15,
               mix('#D9DCEA',PAPER,1-opacity),1)
    stage.rect(x,y,w,7,mix(PURPLE,PAPER,1-opacity),5)
    if w<700:return
    stage.tracking(x+45,y+38,'NODE v08 / TEXT ATTRIBUTES',13,PURPLE,1.5,opacity)
    stage.text(x+43,y+100,'Graph neural networks',40,INK,'bold',opacity)
    stage.text(x+43,y+152,'for semi-supervised classification',32,INK,'regular',opacity)
    stage.line([(x+43,y+222),(x+w-45,y+222)],mix('#DCDFE8',PAPER,1-opacity),1)
    stage.tracking(x+45,y+251,'ABSTRACT',12,GREY,2,opacity)
    prefix='We '
    old='study'
    new='investigate'
    stage.text(x+42,y+292,prefix,31,INK,opacity=opacity)
    wx=x+42+font(31).getlength(prefix)
    change=ease((edited-.07)/.86)
    ww=smooth(font(31).getlength(old),font(31,'bold').getlength(new),change)+13
    if edited>0:
        stage.rect(wx-3,y+283,ww*ease(edited),50,mix('#DDF3E9',PAPER,1-opacity),7)
    stage.text(wx,y+292,old,31,INK,'regular',opacity*(1-change),rise=-16*change)
    stage.text(wx,y+292,new,31,MINT,'bold',opacity*change,rise=17*(1-change))
    end=wx+ww-9
    stage.text(end,y+292,' graph neural networks',31,INK,opacity=opacity)
    stage.text(x+43,y+342,'for semi-supervised learning on',31,INK,opacity=opacity)
    stage.text(x+43,y+392,'a network of linked documents.',31,INK,opacity=opacity)
    stage.text(x+43,y+h-57,'Illustrative edit, not an optimized experimental example.',16,GREY,opacity=opacity)


def response_curve(stage,x,y,w=440,h=181,target=.32,phase=0,edited=0):
    # Category response bars without made-up numerical scores.
    vals=np.array([.69,.2,.14])*(1-edited)+np.array([.16,.79,.10])*edited
    for i,(name,v) in enumerate(zip(['A','TARGET','C'],vals)):
        bx=x+i*(w/3)
        stage.rect(bx,y+h-v*h,80,max(2,v*h),MINT if i==1 else '#C7C7DF',8)
        stage.text(bx+40,y+h+24,name,14,MINT if i==1 else GREY,'mono',anchor='mt')
    stage.line([(x-4,y+h+4),(x+w-45,y+h+4)],'#CDD1DE',1)


def edit(stage,t,duration):
    ui(stage,'02 / SEMANTIC FINGERPRINT')
    stage.text(95,201,'Small text edit.',72,INK,'thin',out(t/1.2),rise=(1-out(t/1.2))*22)
    stage.text(658,201,'A different response.',72,INK,'bold',out((t-.2)/1.2),rise=(1-out((t-.2)/1.2))*22)
    expand=out(t/1.55)
    if expand<.999:
        graph(stage,t,center=(1350,583),zoom=1.82,angle=-.1,tilt=.46,
              selection=1,opacity=1-expand,flow=.5,documents=1,
              target=(-47,36,0),floating=7)
    # A local document expands into the actual content; no slide-style card reveal.
    x=smooth(1320,99,expand);y=smooth(338,338,expand)
    w=smooth(65,902,expand);h=smooth(86,503,expand)
    edited=out((t-duration*.30)/2.4)
    article(stage,x,y,w,h,edited=edited,opacity=expand)
    stage.marker(1174,321,'MODEL RESPONSE / SCHEMATIC',PURPLE,expand)
    response_curve(stage,1190,391,430,197,edited=edited)
    stage.text(1388,669,'CLEAN',17,GREY,'mono',1-edited,'mt')
    stage.text(1388,669,'EDITED',17,MINT,'mono',edited,'mt')
    # Arrow carries a re-encoded feature token from the text to the response.
    arrow(stage,1034,528,1134,528,PURPLE,out((t-1.2)/1.1),2)
    if t>1.4:
        f=((t-1.4)*.55)%1
        stage.sphere(1034+100*f,528,5,MINT,edited)
    # Named transforms and optimization are an editorial strip, not another dashboard.
    step=out((t-1.4)/1.3)
    stage.tracking(116,897,'LLM PROPOSAL',14,PURPLE,1.2,step)
    arrow(stage,357,909,428,909,'#9C9FB4',step)
    stage.tracking(463,897,'SAME TEXT ENCODER',14,PURPLE,1.2,step)
    arrow(stage,775,909,854,909,'#9C9FB4',step)
    stage.tracking(889,897,'SURROGATE SELECTION',14,PURPLE,1.2,step)
    stage.text(1316,767,'Text + structure + target',22,INK,opacity=out((t-2)/1.3),anchor='mt')
    stage.text(1316,807,'Target response + regional drift',18,GREY,opacity=out((t-2.2)/1.3),anchor='mt')


def task_response(stage,x,y,name,before,after,progress=1,zero=False):
    stage.tracking(x,y,name,17,INK,2)
    bx=470
    baseline=y+113
    for side,vals in [(0,before),(1,after)]:
        xx=bx+side*400
        for j,v in enumerate(vals):
            hh=104*v*(1 if side==0 else ease(progress))
            stage.rect(xx+j*77,baseline-hh,54,max(2,hh),MINT if j==1 else '#B8BBD6',5)
        stage.line([(xx-5,baseline+2),(xx+214,baseline+2)],'#D1D5E2',1)
    arrow(stage,bx+276,y+70,bx+335,y+70,'#9C99BC',progress)
    for i,letter in enumerate(['A','TARGET','C']):
        stage.text(bx+i*77+27,baseline+14,letter,11,MINT if i==1 else GREY,'mono',anchor='mt')
        stage.text(bx+400+i*77+27,baseline+14,letter,11,MINT if i==1 else GREY,'mono',anchor='mt')
    stamp=out((progress-.4)/.6)
    xx=1330
    stage.ellipse(xx,y+71,42,42,mix('#E1E3EC' if zero else '#D8F0E7',PAPER,1-stamp))
    stage.text(xx,y+46,'0' if zero else '+1',34,GREY if zero else MINT,'bold',stamp,'mt')
    stage.text(xx+85,y+49,'no change' if zero else 'toward target',20,GREY if zero else MINT,opacity=stamp)
    return (xx,y+71)


def response(stage,t,duration):
    ui(stage,'03 / TASK-WISE EVIDENCE')
    stage.text(95,201,'Read the change.',85,INK,'thin',out(t/1.2),rise=(1-out(t/1.2))*20)
    stage.text(885,224,'Not just the predicted label.',30,GREY,opacity=out((t-.25)/1.2))
    stage.text(570,347,'CLEAN',14,GREY,'mono',anchor='mt')
    stage.text(970,347,'EDITED',14,MINT,'mono',anchor='mt')
    stage.text(1360,347,'EVIDENCE',14,GREY,'mono',anchor='mt')
    for i,(name,before,after) in enumerate([
        ('NODE',[.66,.21,.13],[.14,.76,.10]),
        ('EDGE',[.59,.27,.14],[.18,.72,.10]),
        ('GRAPH',[.64,.22,.14],[.62,.24,.14])]):
        y=396+i*177
        reveal=out((t-duration*.32-i*.7)/1.8)
        task_response(stage,99,y,name,before,after,reveal,i==2)
        if i<2:stage.line([(99,y+145),(1770,y+145)],'#D8DCE7',1)
    stage.marker(104,946,'TASK-LOCAL TARGETS / COMPATIBLE CLASSIFICATION INTERFACE',PURPLE,out((t-2)/1.3))


def fingerprint(stage,x,y,scale=1,progress=1,fill=MINT):
    # A custom visual mark. It is a metaphor for behavioral fingerprints, not a measurement.
    for i in range(8):
        width=(21+i*9)*scale;height=(28+i*11)*scale
        pts=[]
        max_t=math.pi*1.45*sat(progress-i*.045)
        for theta in np.linspace(-math.pi*.18,-math.pi*.18+max_t,100):
            xx=x+width*math.cos(theta)
            yy=y-height*math.sin(theta)
            # Slight offset and open grooves create a recognizable print rather than a target.
            pts.append((xx,yy+i*scale*1.2))
        stage.line(pts,fill,3.3*scale)
    for i in range(3):
        stage.bezier([(x+12*scale+i*9*scale,y+20*scale),
                      (x+8*scale+i*8*scale,y+47*scale),
                      (x-3*scale+i*13*scale,y+74*scale),
                      (x-32*scale+i*14*scale,y+86*scale)],fill,3.3*scale,sat(progress-.35))


def verdict(stage,t,duration):
    ui(stage,'04 / AGGREGATE THE EVIDENCE')
    stage.text(95,201,'Two signals.',84,INK,'thin',out(t/1.2),rise=(1-out(t/1.2))*20)
    stage.text(608,201,'One decision.',84,INK,'bold',out((t-.25)/1.2),rise=(1-out((t-.25)/1.2))*20)
    # Match-cut from the preceding response rows: the evidence marks move, not the slide.
    converge=out((t-1.0)/2.6)
    for i,orig in enumerate([(1330,467),(1330,644),(1330,821)]):
        dest=[(1095,483),(1195,483),(1347,483)][i]
        x,y=lerp(orig,dest,converge)
        positive=i<2
        stage.ellipse(x,y,42,42,'#D8F0E7' if positive else '#E1E3EC')
        stage.text(x,y-25,'+1' if positive else '0',34,MINT if positive else GREY,'bold',1,'mt')
    stage.text(1213,568,'2 positive votes',32,INK,'bold',out((t-2)/1.2),'mt')
    stage.text(1213,618,'Three-task majority threshold: 2',24,GREY,opacity=out((t-2.3)/1.2),anchor='mt')
    fingerprint(stage,405,594,scale=1.75,progress=out((t-.5)/3.0),fill=PURPLE)
    stage.marker(1019,709,'SUPPORTS DATASET USE',MINT,out((t-3)/1.5))
    stage.line([(99,874),(1820,874)],'#CDD2E2',1)
    stage.text(99,915,'Zero evidence',27,GREY,'bold',out((t-duration*.54)/1.2))
    stage.text(337,915,'is not a no-use verdict.',27,INK,opacity=out((t-duration*.54)/1.2))
    stage.text(1814,923,'INSUFFICIENT EVIDENCE / UNCERTAIN',16,GREY,'mono',
               out((t-duration*.64)/1.2),'rt')


def closing(stage,t,duration):
    ui(stage,'TAGFINGER',True)
    fingerprint(stage,982,380,scale=1.0,progress=out(t/1.8),fill=LIME)
    stage.text(960,492,'TAGFinger',153,WHITE,'display',out((t-.2)/1.4),'mt',rise=(1-out((t-.2)/1.4))*20)
    stage.text(960,714,'From graph semantics to ownership evidence.',34,'#B6B9D2',
               opacity=out((t-.65)/1.2),anchor='mt')
    stage.tracking(563,846,'SEMANTIC-PROMPTED / STRUCTURAL-RESILIENT',14,VIOLET,1.8,
                   out((t-.8)/1.5))


SHOTS={'opening':opening,'region':region,'edit':edit,'response':response,'verdict':verdict,'closing':closing}


def subtext(text):
    return text.replace('TAG Finger','TAGFinger').replace('black box','black-box').replace('query-based','query-based')


def stamp(t):
    millis=round(t*1000);h,millis=divmod(millis,3600000);m,millis=divmod(millis,60000);s,millis=divmod(millis,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{millis:03d}'


def synthesize(args):
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    args.assets.mkdir(parents=True,exist_ok=True)
    voice=PiperVoice.load(args.voice_model,config_path=args.voice_config)
    settings=SynthesisConfig(length_scale=.97,noise_scale=.48,noise_w_scale=.57)
    sr=voice.config.sample_rate;pieces=[];cursor=0;timeline=[]
    for i,source in enumerate(STORY['shots']):
        path=args.assets/f'cue_{i}.wav'
        if not path.exists() or args.refresh_audio:
            with wave.open(str(path),'wb') as wav:voice.synthesize_wav(source['cue'],wav,syn_config=settings)
        with wave.open(str(path),'rb') as wav:pcm=np.frombuffer(wav.readframes(wav.getnframes()),np.int16).copy()
        act=np.where(np.abs(pcm.astype(np.int32))>180)[0]
        if len(act)==0:raise RuntimeError('Narration empty')
        pad=round(.05*sr);pcm=pcm[max(0,act[0]-pad):min(len(pcm),act[-1]+pad)]
        start=cursor/sr
        lead=np.zeros(round((.65 if i else 1.1)*sr),np.int16);pieces.append(lead);cursor+=len(lead)
        voice_start=cursor/sr;pieces.append(pcm);cursor+=len(pcm);voice_end=cursor/sr
        tail=np.zeros(round((.85 if i<len(STORY['shots'])-1 else 1.8)*sr),np.int16);pieces.append(tail);cursor+=len(tail)
        scene=dict(source,start=start,end=cursor/sr,voice_start=voice_start,voice_end=voice_end)
        timeline.append(scene)
        print(f'{source["id"]}: {scene["end"]-start:.2f}s',flush=True)
    speech=np.concatenate(pieces).astype(np.float32)/32768
    duration=len(speech)/sr
    # Original low-level score: a slow, warm, filtered chord pad. Not licensed music.
    tt=np.arange(len(speech),dtype=np.float64)/sr
    bed=np.zeros(len(speech),np.float64)
    notes=[130.8128,164.8138,195.9977,246.9417]
    for i,freq in enumerate(notes):
        env=.7+.3*np.sin(2*np.pi*tt/(21+i*7)+i)
        bed+=(np.sin(2*np.pi*freq*tt+.004*np.sin(tt*1.7))*env+
              .13*np.sin(2*np.pi*freq*2*tt))/len(notes)
    fade=np.minimum(np.clip(tt/2,0,1),np.clip((duration-tt)/2.8,0,1))
    # Duck original bed while speech is active. No sound presented as experimental evidence.
    duck=np.ones(len(tt))
    for shot in timeline:
        a,b=round(shot['voice_start']*sr),round(shot['voice_end']*sr)
        duck[a:b]=.65
    bed*=.019*fade*duck
    stereo=np.column_stack((speech*.84+bed,speech*.84+bed*.93))
    events=[(0.3,'swell'),(timeline[1]['start']+.3,'swell'),
            (timeline[2]['start']+2.5,'tick'),(timeline[3]['start']+2.8,'tick'),
            (timeline[4]['start']+3.2,'bell'),(timeline[5]['start']+.2,'swell')]
    for when,kind in events:
        n=round(sr*(.10 if kind=='tick' else 1.1));a=round(when*sr)
        n=min(n,len(tt)-a);x=np.arange(n)/sr
        if kind=='tick':waveform=.022*np.sin(2*np.pi*980*x)*np.exp(-x*42)
        elif kind=='bell':waveform=.014*(np.sin(2*np.pi*659.25*x)+.4*np.sin(2*np.pi*987.77*x))*np.exp(-x*5)
        else:
            rng=np.random.default_rng(a);noise=rng.normal(0,1,n)
            filtered=np.convolve(noise,np.ones(21)/21,'same')
            waveform=filtered*.032*np.sin(np.pi*np.arange(n)/n)**2
        stereo[a:a+n,0]+=waveform;stereo[a:a+n,1]+=waveform
    pcm=np.clip(stereo,-.98,.98)*32767
    with wave.open(str(args.assets/'mix.wav'),'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(sr);wav.writeframes(pcm.astype(np.int16).tobytes())
    meta={'duration':duration,'sample_rate':sr,'shots':timeline,'disclaimer':STORY['disclaimer'],
          'audio':'Local synthetic speech plus original procedural bed and accents'}
    (args.assets/'timeline.json').write_text(json.dumps(meta,indent=2)+'\n')
    srt='\n\n'.join(f'{i+1}\n{stamp(s["voice_start"])} --> {stamp(s["voice_end"])}\n{subtext(s["cue"])}' for i,s in enumerate(timeline))+'\n'
    (args.assets/'TAGFinger_v3_sample.srt').write_text(srt)
    print(f'Duration: {duration:.2f}s')


def render_frame(t,timeline,width=1920,captions=True):
    idx=next((i for i,s in enumerate(timeline['shots']) if t<s['end']),len(timeline['shots'])-1)
    shot=timeline['shots'][idx];local=t-shot['start'];duration=shot['end']-shot['start']
    stage=Stage(width,dark=shot['id'] in ('opening','closing'))
    SHOTS[shot['id']](stage,local,duration)
    # Match dissolves preserve the physical graph / document positions; no blank slide cuts.
    if idx and local<.55:
        previous=timeline['shots'][idx-1]
        previous_stage=Stage(width,dark=previous['id'] in ('opening','closing'))
        previous_duration=previous['end']-previous['start']
        SHOTS[previous['id']](previous_stage,previous_duration,previous_duration)
        stage.im=Image.blend(previous_stage.im,stage.im,ease(local/.55))
    if idx==0 and local<.3:
        stage.im=Image.blend(background(width,True).convert('RGBA'),stage.im,out(local/.3))
    if idx==len(timeline['shots'])-1 and duration-local<.6:
        stage.im=Image.blend(background(width,True).convert('RGBA'),stage.im,out((duration-local)/.6))
    if captions and shot['voice_start']<=t<=shot['voice_end']:
        caption=subtext(shot['cue'])
        # Split long narration into timed clauses, so subtitles stay at one/two lines.
        sentences=[s.strip() for s in re.findall(r'[^.!?]+(?:[.!?]|$)',caption) if s.strip()]
        words=[len(s.split()) for s in sentences];total=sum(words)
        progress=(t-shot['voice_start'])/(shot['voice_end']-shot['voice_start']);cum=0;chosen=sentences[-1]
        for sentence,n in zip(sentences,words):
            cum+=n/total
            if progress<cum:chosen=sentence;break
        line='';lines=[]
        for word in chosen.split():
            candidate=(line+' '+word).strip()
            if line and font(25).getlength(candidate)>1650:
                lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        if len(lines)>2:raise RuntimeError('Subtitle overflow')
        strip=Image.new('RGBA',(width,round((56+27*(len(lines)-1))*stage.s)))
        dark=stage.dark
        ImageDraw.Draw(strip).rectangle((0,0,strip.width,strip.height),fill=(17,19,34,219) if dark else (240,241,246,237))
        stage.paste(strip,0,1032-27*(len(lines)-1),native=True)
        for i,line in enumerate(lines):
            stage.text(960,1041-27*(len(lines)-1)+i*29,line,25,WHITE if dark else INK,anchor='mt')
    return stage.im.convert('RGB')


def stills(args):
    meta=json.loads((args.assets/'timeline.json').read_text());dest=args.assets/'stills';dest.mkdir(exist_ok=True)
    contact=Image.new('RGB',(1536,960),PAPER)
    d=ImageDraw.Draw(contact)
    for i,shot in enumerate(meta['shots']):
        t=shot['start']+(shot['end']-shot['start'])*.77
        image=render_frame(t,meta,args.width,False)
        image.save(dest/f'{i+1:02d}_{shot["id"]}.png')
        image=image.resize((768,432),Image.Resampling.LANCZOS)
        x=(i%2)*768;y=(i//2)*320
        thumb=image.resize((512,288),Image.Resampling.LANCZOS)
        x=(i%3)*512;y=(i//3)*480
        contact.paste(thumb,(x,y))
        d.text((x+17,y+310),f'{i+1:02d} / {shot["id"].upper()}',font=font(22,'bold'),fill=INK)
        d.text((x+17,y+354),f'{shot["start"]:.1f} — {shot["end"]:.1f} s',font=font(17),fill=GREY)
    contact=contact.crop((0,0,1536,890));contact.save(args.assets/'contact.png')
    render_frame(3.5,meta,1920,False).save(OUT/'TAGFinger_v3_sample_poster.jpg',quality=95)
    print('Stills:',dest)


def render(args):
    meta=json.loads((args.assets/'timeline.json').read_text())
    duration=meta['duration'] if args.seconds is None else min(args.seconds,meta['duration'])
    output=args.output or OUT/'TAGFinger_v3_sample.mp4';output.parent.mkdir(exist_ok=True,parents=True)
    total=math.ceil(duration*args.fps);started=time.monotonic()
    ffmpeg=args.ffmpeg
    cmd=[ffmpeg,'-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{args.width}x{round(args.width*9/16)}','-r',str(args.fps),'-i','pipe:0',
         '-i',str(args.assets/'mix.wav'),'-map','0:v','-map','1:a',
         '-vf',f'scale={args.output_width}:-1:flags=lanczos',
         '-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',
         '-c:a','aac','-ar','48000','-b:a','192k',
         '-af','loudnorm=I=-16:TP=-1.5:LRA=9','-t',str(duration),'-movflags','+faststart',
         '-metadata','title='+STORY['title'],'-metadata:s:a:0','language=eng',
         '-metadata','comment=Anonymous visual-direction sample. All method examples are schematic.',str(output)]
    with (args.assets/'render.log').open('w') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for n in range(total):
                image=render_frame(n/args.fps,meta,args.width,True)
                process.stdin.write(image.tobytes())
                if n%(args.fps*5)==0:print(f'{n/args.fps:.1f}/{duration:.1f}s · {n/max(.01,time.monotonic()-started):.1f} fps',flush=True)
            process.stdin.close();result=process.wait()
            if result:raise RuntimeError('FFmpeg failed; inspect render.log')
        except BaseException:
            process.kill();process.wait();raise
    print(f'Wrote {output} in {time.monotonic()-started:.1f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('audio','stills','render'))
    p.add_argument('--assets',type=Path,default=ROOT/'assets')
    p.add_argument('--voice-model',default='/tmp/tagfinger-en_US-ljspeech-high.onnx')
    p.add_argument('--voice-config',default='/tmp/tagfinger-ljspeech-config.json')
    p.add_argument('--refresh-audio',action='store_true')
    p.add_argument('--width',type=int,default=2560)
    p.add_argument('--output-width',type=int,default=1920)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--seconds',type=float)
    p.add_argument('--ffmpeg',default='/tmp/tagfinger-ffmpeg')
    p.add_argument('--output',type=Path)
    args=p.parse_args();{'audio':synthesize,'stills':stills,'render':render}[args.mode](args)


if __name__=='__main__':main()
