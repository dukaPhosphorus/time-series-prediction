# coding: utf-8

from math import log, pow, sqrt
import numpy as np
from scipy.stats import norm, cauchy
from numpy.random import uniform, multivariate_normal
from multiprocessing import Pool
import matplotlib.pyplot as plt


class ParticleFilter:    
    log_likelihood = 0.0 # 対数尤度
    TIME = 1
    PR=8 # unmber of processing

    def __init__(self, PARTICLES_NUM, k=1, ydim=1, sys_pdim=1, ob_pdim=1, sh_parameters=[0.01, 0.35]):
        self.nois_sh_parameters = sh_parameters # nu:システムノイズの位置超々パラメタ , xi:観測ノイズの位置超々パラメタ
        pdim = sys_pdim+ob_pdim
        self.PARTICLES_NUM = PARTICLES_NUM # 粒子数
        self.TEETH_OF_COMB = np.arange(0, 1, float(1.0)/self.PARTICLES_NUM)
        self.weights = np.zeros((ydim, self.PARTICLES_NUM), dtype=np.float64)
        self.particles = np.zeros((k*ydim+pdim ,self.PARTICLES_NUM), dtype=np.float64)
        self.predicted_particles = np.zeros((k*ydim+pdim , self.PARTICLES_NUM), dtype=np.float64)
        np.random.seed(555)
        self.predicted_value = []
        self.filtered_value = []
        self.sys_nois = []
        self.ob_nois = []
        self.LSM = np.zeros(ydim) # ２乗誤差

        self.F, self.G, self.H= self.FGHset(k, ydim, pdim)
        self.k = k
        self.ydim = ydim
        self.pdim = pdim
        self.sys_pdim = sys_pdim
        self.ob_pdim = ob_pdim

    def init_praticles_distribution(self, P, r):
        """initialize particles
        x_0|0
        tau_0|0
        sigma_0|0
        """
        data_particles = multivariate_normal([1]*self.ydim*self.k,
                            np.eye(self.ydim*self.k)*10, self.PARTICLES_NUM).T
        param_particles = np.zeros((self.pdim, self.PARTICLES_NUM))
        for i in xrange(self.pdim):
            param_particles[i,:] = uniform(P-r, P+r, self.PARTICLES_NUM)
        self.particles = np.vstack((data_particles, param_particles))

    def get_system_noise(self):
        """v_t vector"""
        data_noise = np.zeros((self.ydim, self.PARTICLES_NUM), dtype=np.float64)
        for i in xrange(self.ydim):
            data_noise[i,:] = cauchy.rvs(loc=[0]*self.PARTICLES_NUM, scale=np.power(10,self.particles[self.ydim]),
                                    size=self.PARTICLES_NUM)
        data_noise[data_noise==float("-inf")] = -1e308
        data_noise[data_noise==float("inf")] = 1e308

        parameter_noises = np.zeros((self.pdim, self.PARTICLES_NUM), dtype=np.float64)
        for i in xrange(self.pdim):
            parameter_noises[i,:] = cauchy.rvs(loc=0, scale=self.nois_sh_parameters[i], size=self.PARTICLES_NUM)
        return np.vstack((data_noise, parameter_noises))

    def calc_pred_particles(self):
        """calculate system function
        x_t|t-1 = F*x_t-1 + Gv_t
        """
        return self.F.dot(self.particles) + self.G.dot(self.get_system_noise()) # linear non-Gaussian  

    def calc_particles_weight(self,y):
        """calculate fitness probabilities between observation value and predicted value
        w_t
        """
        locs = self.calc_pred_particles()
        self.predicted_particles = locs
        scale=np.power(10,locs[-1])
        scale[scale==0] = 1e-308

        # 多変量の場合などは修正が必要
        self.weights = cauchy.pdf( np.array([y]*self.PARTICLES_NUM) - self.H.dot(locs), loc=[0]*self.PARTICLES_NUM,
                                scale=scale, size=self.PARTICLES_NUM).flatten()

    def calc_likelihood(self):
        """calculate likelihood at that point
        p(y_t|y_1:t-1)
        """
        res = np.sum(self.weights)/self.PARTICLES_NUM
        self.log_likelihood += log(res)

    def normalize_weights(self):
        """wtilda_t"""
        self.weights = self.weights/np.sum(self.weights)

    def resample(self,y):
        """x_t|t"""
        self.normalize_weights()

        self.memorize_predicted_value()

        # accumulate weight
        cum = np.cumsum(self.weights)

        # create roulette pointer 
        base = uniform(0,float(1.0)/self.PARTICLES_NUM)
        pointers = self.TEETH_OF_COMB + base

        # select particles
        selected_idx = [np.where(cum>=p)[0][0] for p in pointers]
        """
        pool = Pool(processes=self.PR)
        selected_idx = pool.map(get_slected_particles_idx, ((cum,p) for p in pointers))
        pool.close()
        pool.join()     
        """

        self.particles = self.predicted_particles[:,selected_idx]
        self.memorize_filtered_value(selected_idx, y)

    def memorize_predicted_value(self):
        predicted_value = np.sum(self.predicted_particles*self.weights, axis=1)[0]
        self.predicted_value.append(predicted_value)

    def memorize_filtered_value(self, selected_idx, y):
        filtered_value = np.sum(self.particles*self.weights[selected_idx] , axis=1) \
                            /np.sum(self.weights[selected_idx])
        self.filtered_value.append(filtered_value[:self.ydim])
        self.sys_nois.append(np.power(10,filtered_value[self.ydim:self.ydim+self.sys_pdim]))
        self.ob_nois.append(np.power(10,filtered_value[self.ydim+self.sys_pdim:]))
        self.calculate_LSM(y,filtered_value[:self.ydim])

    def calculate_LSM(self,y,filterd_value):
        self.LSM += pow(y-filterd_value,2)

    def forward(self,y):
        """compute system model and observation model"""
        print 'calculating time at %d' % self.TIME
        self.calc_pred_particles()
        self.calc_particles_weight(y)
        self.calc_likelihood()
        self.resample(y)
        self.TIME += 1

    def FGHset(self, k, vn_y, n_h_parameters):
        """状態空間表現の行列設定
        vn_y:入力ベクトルの次元
        n_h_parameters:ハイパーパラメタ数
        k：階差
        """
        G_upper_block = np.zeros((k*vn_y, vn_y+n_h_parameters))
        G_lower_block = np.zeros((n_h_parameters, vn_y+n_h_parameters))
        G_lower_block[-n_h_parameters:, -n_h_parameters:] = np.eye(n_h_parameters)
        G_upper_block[:vn_y, :vn_y] = np.eye(vn_y)
        G = np.vstack( (G_upper_block, G_lower_block) )

        H = np.hstack( (np.eye(vn_y), 
                            np.zeros((vn_y, vn_y*(k-1)+n_h_parameters))
                        ) )

        # トレンドモデルのブロック行列の構築
        F_upper_block = np.zeros((k*vn_y, k*vn_y+n_h_parameters))
        F_lower_block = np.zeros((n_h_parameters, k*vn_y+n_h_parameters))
        F_lower_block[-n_h_parameters:, -n_h_parameters:] = np.eye(n_h_parameters)
        if k==1:
            F_upper_block[:vn_y, :vn_y] = np.eye(vn_y)
        elif k==2:
            F_upper_block[:vn_y, :vn_y] = np.eye(vn_y)*2
            F_upper_block[:vn_y, vn_y:k*vn_y] = np.eye(vn_y)*-1
            F_upper_block[vn_y:k*vn_y, :vn_y] = np.eye(vn_y)
        F = np.vstack((F_upper_block, F_lower_block))

        return F, G, H

def get_slected_particles_idx((cum,p)):
    """multiprocessing function"""
    try:
        return np.where(cum>=p)[0][0]
    except Exception, e:
        import sys
        import traceback
        sys.stderr.write(traceback.format_exc())    

if __name__=='__main__':
    n_particle = 1000
    nu=0.01
    xi=0.35
    pf = ParticleFilter(n_particle, k=1, ydim=1, sys_pdim=1, ob_pdim=1, sh_parameters=[nu, xi])
    pf.init_praticles_distribution(0, 8) # P, r

    data = np.hstack((norm.rvs(0,1,size=20),norm.rvs(10,1,size=60),norm.rvs(-30,0.5,size=20)))

    for d in data:
        pf.forward(d)
    print 'log likelihood:', pf.log_likelihood
    print 'LSM:', pf.LSM

    rng = range(100)
    plt.plot(rng,data,label=u"training data")
    plt.plot(rng,pf.predicted_value,label=u"predicted data")
    plt.plot(rng,pf.filtered_value,label=u"filtered data")
#    plt.plot(rng,pf.sys_nois,label=u"system noise hyper parameter")
#    plt.plot(rng,pf.ob_nois,label=u"observation noise hyper parameter")
    plt.xlabel('TIME',fontsize=18)
    plt.ylabel('Value',fontsize=18)    
    plt.legend(loc = 'upper left') 
    plt.show()
