from pylab import *
from scipy import *
from matplotlib import cm
import matplotlib.pyplot as plt




X = 1          #単位は[m]
Y = 1

dx = 0.04    #空間離散化幅[m]
dy = 0.04
dt = 0.00002      #時間離散化幅[s]

Ro = 1.21        #空気の密度
C = 343          #音速[m/s]
K = Ro*C*C       #体積弾性率

step = 1500 #ステップ数(×dtで秒)
f = 122          #音源の周波数(Hz)
Q = zeros((step,1), "float64")#音源
for m in range(step) :
    Q[m] = sin(2*pi*f*m*dt) 

X1 = int(X/dx)   #横メッシュの個数
Y1 = int(Y/dy)   #縦メッシュの個数

X2 = int(X1*0.65)#共鳴管の長さ
Y2 = int(0.15*Y1)#共鳴管の幅の半分の長さ

P1 = zeros((X1,Y1),"float64")#1個前の音圧を格納する配列
P2 = zeros((X1,Y1),"float64")#次の音圧を格納する配列

Ux1 = zeros((X1+1,Y1),"float64")#x方向の粒子速度
Ux2 = zeros((X1+1,Y1),"float64")

Uy1 = zeros((X1,Y1+1),"float64")#y方向の粒子速度
Uy2 = zeros((X1,Y1+1),"float64")

Y3 = int(Y1/2)

fig = figure()

print("X1=",X1, ", Y1=",Y1, ", X2=",X2, ", Y2=",Y2, ", Y3=",Y3 )

upwall, downwall = np.zeros(X1), np.zeros(X1)
for x in range(X1):
    if (x>2 and x<3+X2):
        upwall[x] = Y3-Y2
        downwall[x] = Y3+Y2-1
    else:
        upwall[x] = nan
        downwall[x] = nan        

for n in range(step) :
    plt.cla()
    cont=contour(P1.T,  8, Vmax=1,colors=['black'])
    cont.clabel(fmt='%1.1f', fontsize=12)
    imshow(P1.T)
    xlim(0,X1-1)     #x軸設定
    ylim(0,Y1-1)     #y軸設定
    xlabel("X [sample]")
    ylabel("Y [sample]")
    plt.plot(upwall, label='upwall', color='red')
    plt.plot(downwall, label='downwall', color='red')
    plt.pause(0.03)
    
    P1[X1-1,Y3] = Q[n]  #音源の位置は右端の中央
    P1[X1-1,Y3-1] = Q[n]
    P1[X1-1,Y3+1] = Q[n]
    
    for x in range(X1-1):       #運動方程式(x)
        for y in range(Y1):
            if (x==0 or x== X1-2):
                Ux2[x+1,y]=Ux1[x+1,y]-dt/Ro/dx*(P1[x+1,y]-P1[x,y])
            else:
                Ux2[x+1,y]=Ux1[x+1,y]-dt/Ro/dx*(-1*P1[x+2,y]/24 + 9*P1[x+1,y]/8 \
                                            -9*P1[x,y]/8 + 1*P1[x-1,y]/24)


            if ((x > 2) and (x < 3 + X2) and (y == (Y3-Y2) or y == (Y3+Y2))) :
                Ux2[x,y] = 0.0
            if (x==3 or x == X2+3) and (y>Y3-Y2-1) and (y<Y3+Y2+1):
                Ux2[x,y] = 0.0
                    
    for x in range(X1):         #運動方程式(y)
        for y in range(Y1-1):
            if (y==0 or y== Y1-2):
                Uy2[x,y+1]=Uy1[x,y+1]-dt/Ro/dx*(P1[x,y+1]-P1[x,y])
            else:
                Uy2[x,y+1]=Uy1[x,y+1]-dt/Ro/dx*(-1*P1[x,y+2]/24 + 9*P1[x,y+1]/8 \
                                            -9*P1[x,y]/8 + 1*P1[x,y-1]/24)
                 
            if ((x > 2) and (x < 3 + X2) and (y == (Y3-Y2) or y == (Y3+Y2))) :
                Uy2[x,y] = 0.0
            if (x==3 or x==X2+3) and (y>Y3-Y2-1) and (y<Y3+Y2+1):
                Uy2[x,y] = 0.0
                    
    for x in range(X1):         #連続の式
        for y in range(Y1):
            if (x==0 or x==X1-1 or y==0 or y==Y1-1):
                P2[x,y] = P1[x,y]-K*dt/dx*(Ux2[x+1,y]-Ux2[x,y]) \
                -K*dt/dy*(Uy2[x,y+1]-Uy2[x,y])
            else:
                P2[x,y] = P1[x,y]-K*dt/dx*(-1*Ux2[x+2,y]/24 + 9*Ux2[x+1,y]/8    \
                                       -9*Ux2[x,y]/8 + 1*Ux2[x-1,y]/24)     \
                             -K*dt/dy*(-1*Uy2[x,y+2]/24 + 9*Uy2[x,y+1]/8    \
                                       -9*Uy2[x,y]/8 + 1*Uy2[x,y-1]/24)

    P2[3][Y3-Y2]= P2[3][Y3-Y2+1]
            
    P1,P2=P2,P1  #各変数の更新
    Ux1,Ux2=Ux2,Ux1
    Uy1,Uy2=Uy2,Uy1

figure()
contourf(P1.T,aspect="equal")
cont=contour(P1.T,  8, Vmax=1,colors=['black'])
cont.clabel(fmt='%1.1f', fontsize=12)
imshow(P1.T)    
xlim(0,X1-1)     #x軸設定
ylim(0,Y1-1)     #y軸設定
xlabel("X [sample]")
ylabel("Y [sample]")
plt.plot(upwall, label='upwall', color='red')
plt.plot(downwall, label='downwall', color='red')
