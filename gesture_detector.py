"""
gesture_fx.py  —  Dragon Ball / Naruto Real Visual Effects
Controls: Q / ESC = quit   S = screenshot
"""

import cv2
import numpy as np
import time
import random
import math
from collections import deque

from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ── MediaPipe ────────────────────────────────────────────────────────────────
detector = Hands(max_num_hands=2,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.65)

# ── Geometry helpers ──────────────────────────────────────────────────────────
def px(lm, w, h):          return (int(lm.x * w), int(lm.y * h))
def center(lms, w, h):
    ids = [0, 1, 5, 9, 13, 17]
    return (int(np.mean([lms[i].x for i in ids]) * w),
            int(np.mean([lms[i].y for i in ids]) * h))
def fingertip(lms, idx, w, h):
    return px(lms[[4,8,12,16,20][idx]], w, h)
def wrist_gap(a, b):
    return math.hypot(a[0].x-b[0].x, a[0].y-b[0].y)

# ── Finger state ──────────────────────────────────────────────────────────────
def fingers(lm):
    tips=[4,8,12,16,20]; pips=[3,6,10,14,18]
    f = [lm[4].x < lm[3].x]
    f += [lm[tips[i]].y < lm[pips[i]].y for i in range(1,5)]
    return f

# ── Gesture classifier ────────────────────────────────────────────────────────
def classify(all_lm):
    if not all_lm: return None
    if len(all_lm) == 1:
        f = fingers(all_lm[0])
        if not any(f):                           return "RASENGAN"
        if all(f):                               return "SPIRIT_BOMB"
        if f==[False,True,True,False,False]:     return "SHADOW_CLONE"
        if f==[False,True,False,False,False]:    return "CHIDORI"
        if f==[False,False,True,True,True]:      return "FIRE_STYLE"
        if f==[True,False,False,False,False]:    return "POWER_UP"
    if len(all_lm) == 2:
        f1,f2 = fingers(all_lm[0]),fingers(all_lm[1])
        d = wrist_gap(all_lm[0], all_lm[1])
        if not any(f1) and not any(f2):
            return "KAMEHAMEHA" if d < 0.22 else "KAMEHAMEHA_CHARGE"
        if all(f1) and all(f2):
            return "FINAL_FLASH" if d < 0.22 else "BIG_BANG"
        if (not any(f1) and all(f2)) or (all(f1) and not any(f2)):
            return "RASENGAN_FULL"
    return None

class Smoother:
    def __init__(self, win=6, hold=10):
        self.q=deque(maxlen=win); self.held=None; self.hold=0; self.H=hold
    def update(self, g):
        self.q.append(g)
        votes={}
        for v in self.q:
            if v: votes[v]=votes.get(v,0)+1
        if votes:
            best=max(votes,key=votes.get)
            if votes[best] >= len(self.q)//2+1:
                self.held=best; self.hold=self.H; return best
        if self.hold>0: self.hold-=1; return self.held
        self.held=None; return None

# ── Particle system ───────────────────────────────────────────────────────────
class P:
    __slots__=('x','y','vx','vy','r','g','b','life','mlife','sz')
    def __init__(self,x,y,vx,vy,col,life,sz):
        self.x=float(x); self.y=float(y)
        self.vx=float(vx); self.vy=float(vy)
        self.r,self.g,self.b=col
        self.life=life; self.mlife=life; self.sz=sz

class Particles:
    def __init__(self, cap=300): self.ps=[]; self.cap=cap

    def emit(self, x, y, col, n=5, spread=30, spd=3, life=25, sz=4, up=False):
        for _ in range(n):
            a=random.uniform(0,2*math.pi)
            s=random.uniform(.5,spd)
            vx=math.cos(a)*s*(spread/30)
            vy=(-abs(math.sin(a)*s) if up else math.sin(a)*s)*(spread/30)
            self.ps.append(P(x,y,vx,vy,col,
                             random.randint(life//2,life),
                             random.randint(max(1,sz-1),sz+2)))
        if len(self.ps)>self.cap: self.ps=self.ps[-self.cap:]

    def spiral(self, cx, cy, col, t, r=50, n=3):
        for i in range(n):
            a=t*4+i*(2*math.pi/n)
            x=cx+r*math.cos(a); y=cy+r*math.sin(a)
            vx=-math.sin(a)*2; vy=math.cos(a)*2
            self.ps.append(P(x,y,vx,vy,col,15,4))

    def tick(self, frame):
        alive=[]
        for p in self.ps:
            p.x+=p.vx; p.y+=p.vy; p.vy+=0.15; p.life-=1
            if p.life>0:
                a=p.life/p.mlife
                sz=max(1,int(p.sz*a))
                col=(int(p.b*a),int(p.g*a),int(p.r*a))
                cv2.circle(frame,(int(p.x),int(p.y)),sz,col,-1)
                alive.append(p)
        self.ps=alive

# ── Drawing primitives ────────────────────────────────────────────────────────
def lightning(dst, a, b, col, w=2, jit=15, depth=3):
    if depth==0:
        cv2.line(dst, a, b, col, w, cv2.LINE_AA); return
    mx=(a[0]+b[0])//2+random.randint(-jit,jit)
    my=(a[1]+b[1])//2+random.randint(-jit,jit)
    m=(mx,my)
    lightning(dst,a,m,col,w,jit,depth-1)
    lightning(dst,m,b,col,w,jit,depth-1)

def glow(frame, layer, ks=55):
    ks=ks|1
    blurred=cv2.GaussianBlur(layer,(ks,ks),0)
    result=np.clip(frame.astype(np.int32)+blurred,0,255).astype(np.uint8)  # ✅
    frame[:]=result

# ── Per-gesture effects ───────────────────────────────────────────────────────
def fx_rasengan(frame, gl, lm, w, h, t, pts):
    cx,cy=center(lm,w,h)
    col=(255,60,220)
    pts.spiral(cx,cy,col,t,r=45)
    pts.spiral(cx,cy,(160,60,255),t+math.pi,r=28)
    for r,a in [(55,.5),(38,.8),(22,1.),(12,1.)]:
        c=tuple(int(v*a) for v in col)
        cv2.circle(gl,(cx,cy),r,c,-1)
    rr=int(62+8*math.sin(t*6))
    cv2.circle(gl,(cx,cy),rr,col,3)

def fx_chidori(frame, gl, lm, w, h, t, pts):
    tip=fingertip(lm,1,w,h)
    col=(50,240,255)
    for _ in range(5):
        e=(tip[0]+random.randint(-90,90), tip[1]+random.randint(-90,90))
        lightning(gl,tip,e,col,w=2,jit=14)
        lightning(frame,tip,e,(180,180,255),w=1,jit=8)
    cv2.circle(gl,tip,28,col,-1)
    cv2.circle(gl,center(lm,w,h),18,(100,200,255),-1)
    pts.emit(*tip,col,n=3,spd=5,life=10,sz=2)

def fx_spirit_bomb(frame, gl, lm, w, h, t, pts):
    pl=center(lm,w,h)
    bx,by=pl[0],pl[1]-130
    pulse=0.85+0.15*math.sin(t*3)
    rad=int(65*pulse)
    ci=(255,255,255); co=(120,190,255)
    for i in range(5):
        a=t*2+i*2*math.pi/5
        dist=random.uniform(90,210)
        px2=bx+dist*math.cos(a); py2=by+dist*math.sin(a)
        vx=(bx-px2)*.05; vy=(by-py2)*.05
        pts.ps.append(P(px2,py2,vx,vy,co,20,3))
    for r,c in [(rad+35,co),(rad+18,co),(rad,ci),(rad-18,ci)]:
        if r>0: cv2.circle(gl,(bx,by),r,c,-1)
    cv2.line(gl,pl,(bx,by),(150,200,255),2)

def fx_kamehameha_charge(frame, gl, lms, w, h, t, pts):
    c1=center(lms[0],w,h); c2=center(lms[1],w,h)
    mid=((c1[0]+c2[0])//2,(c1[1]+c2[1])//2)
    col=(255,130,0)
    for _ in range(3): lightning(gl,c1,c2,col,w=3,jit=22)
    r=int(45*(0.7+0.3*math.sin(t*8)))
    cv2.circle(gl,mid,r,col,-1)
    cv2.circle(gl,mid,r+18,(200,80,0),2)
    pts.emit(*mid,col,n=5,spread=20,spd=3,life=20)

def fx_kamehameha(frame, gl, lms, w, h, t, pts):
    c1=center(lms[0],w,h); c2=center(lms[1],w,h)
    mid=((c1[0]+c2[0])//2,(c1[1]+c2[1])//2)
    col=(255,130,0)
    bdir=-1 if mid[0]>w//2 else 1
    bx=0 if bdir==-1 else w
    bw=int(45+12*math.sin(t*10))
    cv2.rectangle(gl,(mid[0],mid[1]-bw),(bx,mid[1]+bw),col,-1)
    cv2.rectangle(gl,(mid[0],mid[1]-bw//3),(bx,mid[1]+bw//3),(255,210,120),-1)
    for c in [c1,c2]: cv2.circle(gl,c,38,col,-1)
    pts.emit(*mid,col,n=8,spread=55,spd=9,life=20)

def fx_final_flash(frame, gl, lms, w, h, t, pts):
    c1=center(lms[0],w,h); c2=center(lms[1],w,h)
    mid=((c1[0]+c2[0])//2,(c1[1]+c2[1])//2)
    col=(0,255,255)
    for i in range(4):
        r=int((50+i*28)*(0.8+0.2*math.sin(t*5+i)))
        cv2.circle(gl,mid,r,col,2+i)
    cv2.circle(gl,mid,65,col,-1)
    bw=int(32+6*math.sin(t*12))
    cv2.rectangle(gl,(mid[0],mid[1]-bw),(w,mid[1]+bw),col,-1)
    cv2.rectangle(gl,(mid[0],mid[1]-bw//3),(w,mid[1]+bw//3),(255,255,255),-1)
    pts.emit(*mid,col,n=7,spd=7,life=20)

def fx_shadow_clone(frame, gl, lm, w, h, t, pts):
    col=(0,130,255)
    for ox,oy in [(-45,-8),(45,-8)]:
        M=np.float32([[1,0,ox],[0,1,oy]])
        ghost=cv2.warpAffine(frame,M,(w,h))
        cv2.addWeighted(frame,1.,ghost,.28,0,frame)
    pl=center(lm,w,h)
    r=int(52*(0.8+0.2*math.sin(t*5)))
    cv2.circle(gl,pl,r,col,3)
    pts.emit(*pl,col,n=3,spd=3,life=20)

def fx_fire(frame, gl, lm, w, h, t, pts):
    cols=[(0,50,255),(0,120,255),(0,200,255)]
    for i in range(1,5):
        ft=fingertip(lm,i,w,h)
        c=random.choice(cols)
        pts.emit(*ft,c,n=4,spread=18,spd=4,life=22,sz=5)
        cv2.circle(gl,ft,16,c,-1)

def fx_big_bang(frame, gl, lms, w, h, t, pts):
    c1=center(lms[0],w,h); c2=center(lms[1],w,h)
    mid=(w//2,(c1[1]+c2[1])//2)
    col=(220,255,255)
    for i in range(4):
        r=int(((t*90)%320)+i*80)
        a=max(0.,1.-r/320)
        if a>0:
            c2c=tuple(int(v*a) for v in col)
            cv2.circle(gl,mid,r,c2c,3)
    if random.random()<.3:
        pts.emit(*mid,col,n=9,spread=65,spd=9,life=25)

def fx_power_up(frame, gl, lm, w, h, t, pts):
    pl=center(lm,w,h)
    col=(0,255,60)
    for i in range(4):
        r=int((32+i*22)*(0.88+0.12*math.sin(t*6+i*.5)))
        c=tuple(int(v*(1-i*.18)) for v in col)
        cv2.circle(gl,pl,r,c,2)
    pts.emit(pl[0],pl[1]+25,col,n=4,spread=28,spd=4,life=30,up=True)

def fx_rasengan_full(frame, gl, lms, w, h, t, pts):
    c1=center(lms[0],w,h); c2=center(lms[1],w,h)
    col=(255,60,220)
    pts.spiral(c1[0],c1[1],col,t,r=48)
    pts.spiral(c2[0],c2[1],(160,80,255),t+math.pi,r=35)
    for c in [c1,c2]:
        for r in [58,42,26]: cv2.circle(gl,c,r,col,-1)
    lightning(gl,c1,c2,col,w=4,jit=18)

# ── Dispatch ──────────────────────────────────────────────────────────────────
DISPATCH = {
    "RASENGAN":          (fx_rasengan,        1),
    "CHIDORI":           (fx_chidori,         1),
    "SPIRIT_BOMB":       (fx_spirit_bomb,     1),
    "SHADOW_CLONE":      (fx_shadow_clone,    1),
    "FIRE_STYLE":        (fx_fire,            1),
    "POWER_UP":          (fx_power_up,        1),
    "KAMEHAMEHA_CHARGE": (fx_kamehameha_charge,2),
    "KAMEHAMEHA":        (fx_kamehameha,      2),
    "FINAL_FLASH":       (fx_final_flash,     2),
    "BIG_BANG":          (fx_big_bang,        2),
    "RASENGAN_FULL":     (fx_rasengan_full,   2),
}

def apply(gesture, frame, all_lm, w, h, t, pts):
    if gesture not in DISPATCH: return frame
    fn, needed = DISPATCH[gesture]
    if len(all_lm) < needed: return frame
    gl = np.zeros_like(frame)
    fn(frame, gl, all_lm if needed==2 else all_lm[0], w, h, t, pts)
    glow(frame, gl, ks=51)
    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth = Smoother(win=5, hold=10)
    pts    = Particles(cap=300)
    fps_q  = deque(maxlen=30)
    t_last = time.time()
    shot_n = 0

    print("✅ Running — Q/ESC quit  S screenshot")
    print("Gestures: fist=RASENGAN  ✌️=SHADOW CLONE  ☝️=CHIDORI")
    print("          🖐=SPIRIT BOMB  👌=FIRE  👍=POWER UP")
    print("  2 hands: fists close=KAMEHAMEHA  open close=FINAL FLASH")
    print("           fists apart=CHARGE  open apart=BIG BANG")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # FPS
        now=time.time(); fps_q.append(now-t_last); t_last=now
        fps=1./(sum(fps_q)/len(fps_q)) if fps_q else 0

        # MediaPipe
        res  = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        all_lm=[]
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, HAND_CONNECTIONS)
                all_lm.append(list(hl.landmark))

        gesture = smooth.update(classify(all_lm))
        frame   = apply(gesture, frame, all_lm, w, h, now, pts)
        pts.tick(frame)

        # Minimal HUD
        cv2.putText(frame,f"FPS {fps:.0f}",(10,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(180,180,180),1)
        if gesture:
            label=gesture.replace("_"," ")
            cv2.putText(frame,label,(10,h-20),
                        cv2.FONT_HERSHEY_DUPLEX,1.3,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow("Gesture FX", frame)
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),27): break
        elif key==ord('s'):
            f=f"shot_{shot_n:03d}.jpg"; cv2.imwrite(f,frame)
            print(f"Saved {f}"); shot_n+=1

    cap.release(); cv2.destroyAllWindows(); detector.close()
    print("Done")

if __name__=="__main__":
    main()