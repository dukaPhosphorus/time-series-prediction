# coding: utf-8
# 1期先予測→180期先予測
# 180期予測スタート時に180期予測→180期先の長期予測と1期先予測+フィルタ修正を同時進行
# つまり180期先予測ができたかどうかのFBはなく、毎回長期予測やりっぱなし方式
# 180期予測中はずっと同じ観測値を使用

from cgitb import small
from math import log, pow, sqrt
import numpy as np
from scipy.stats import norm, cauchy
from numpy.random import uniform, multivariate_normal
from multiprocessing import Pool
import matplotlib.pyplot as plt
import dataset
import gc
import copy

class ParticleFilter:    
    log_likelihood = 0.0 # 対数尤度
    TIME = 1
    PR=8 # number of processing

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
        self.long_term_predicted_value = []
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
        self.selected_idx = []

    def init_praticles_distribution(self, P, r):
        """initialize particles
        x_0|0
        tau_0|0
        sigma_0|0
        """
        data_particles = multivariate_normal([1]*self.ydim*self.k,
                            np.eye(self.ydim*self.k)*10, self.PARTICLES_NUM).T
        param_particles = np.zeros((self.pdim, self.PARTICLES_NUM))
        for i in range(self.pdim):
            param_particles[i,:] = uniform(P-r, P+r, self.PARTICLES_NUM)
        self.particles = np.vstack((data_particles, param_particles))

    def get_system_noise(self):
        """v_t vector"""
        data_noise = np.zeros((self.ydim*self.k, self.PARTICLES_NUM), dtype=np.float64)
        for i in range(self.ydim):
            data_noise[i,:] = cauchy.rvs(loc=[0]*self.PARTICLES_NUM, scale=np.power(10,self.particles[self.ydim]),
                                    size=self.PARTICLES_NUM)
        data_noise[data_noise==float("-inf")] = -1e308
        data_noise[data_noise==float("inf")] = 1e308

        parameter_noises = np.zeros((self.pdim, self.PARTICLES_NUM), dtype=np.float64)
        for i in range(self.pdim):
            parameter_noises[i,:] = cauchy.rvs(loc=0, scale=self.nois_sh_parameters[i], size=self.PARTICLES_NUM)
        return np.vstack((data_noise, parameter_noises))

    def calc_pred_particles(self,particles_array=None):
        """calculate system function
        x_t|t-1 = F*x_t-1 + Gv_t
        """
        if particles_array is None:
            particles_array = self.particles
        noise = self.get_system_noise()
        return self.F.dot(particles_array) + self.G.dot(noise) # linear non-Gaussian  

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
                                scale=scale).flatten()#, size=self.PARTICLES_NUM).flatten()

    def calc_likelihood(self):
        """calculate likelihood at that point
        p(y_t|y_1:t-1)
        """
        res = np.sum(self.weights)/self.PARTICLES_NUM
        self.log_likelihood += log(res)

    def normalize_weights(self):
        """wtilda_t"""
        self.weights = self.weights/np.sum(self.weights)

    def resample(self,y,predicted_array,filtered_array,sys_nois_array,ob_nois_array):
        """x_t|t 層化抽出方式のリサンプリング"""
        self.normalize_weights()

        if predicted_array is None:
            predicted_array = self.predicted_value
        self.memorize_predicted_value(self.predicted_particles, predicted_array)

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
        self.selected_idx = selected_idx[:]
        self.particles = self.predicted_particles[:,selected_idx]
        if filtered_array is None:
            filtered_array = self.filtered_value
        if sys_nois_array is None:
            sys_nois_array = self.sys_nois
        if ob_nois_array is None:
            ob_nois_array = self.ob_nois
        self.memorize_filtered_value(selected_idx, y, filtered_array, sys_nois_array, ob_nois_array)

    def memorize_predicted_value(self, predicted_part, predicted_array):
        predicted_value = np.sum(predicted_part*self.weights, axis=1)[0]
        predicted_array.append(predicted_value)

    def memorize_filtered_value(self, selected_idx, y, filtered_array, sys_nois_array, ob_nois_array):
        filtered_value = np.sum(self.particles*self.weights[selected_idx] , axis=1) \
                            /np.sum(self.weights[selected_idx])
        filtered_array.append(filtered_value[:self.ydim*self.k])
        sys_nois_array.append(np.power(10,filtered_value[self.ydim*self.k:self.ydim*self.k+self.sys_pdim]))
        ob_nois_array.append(np.power(10,filtered_value[self.ydim*self.k+self.sys_pdim:]))
        self.calculate_LSM(y,filtered_value[self.ydim*self.k])

    def calculate_LSM(self,y,filterd_value):
        self.LSM += pow(y-filterd_value,2)

    def forward(self,y):
        """compute system model and observation model"""
        print (f'calculating time at {self.TIME}')
        if self.TIME < 50:
            # self.calc_pred_particles()
            self.calc_particles_weight(y)
            self.calc_likelihood()
            self.resample(y,predicted_array=None,filtered_array=None,sys_nois_array=None,ob_nois_array=None)
        if self.TIME == 50:
            self.long_term_predicted_value = self.predicted_value[:]
        if 50 <= self.TIME:
            small_ltpv = self.predicted_value[:]
            small_ltpp = self.predicted_particles
            small_ltfv = self.filtered_value[:]
            small_ltp = self.particles[:]
            # small_ltsn = self.sys_nois[:]
            # small_lton = self.ob_nois[:]

            # self.calc_pred_particles()
            self.calc_particles_weight(y)
            self.calc_likelihood()
            self.resample(y,predicted_array=None,filtered_array=None,sys_nois_array=None,ob_nois_array=None)
            for tau in range(5):
                locs = self.calc_pred_particles(small_ltp)
                small_ltpp = locs
                self.memorize_predicted_value(small_ltpp, small_ltpv)
                small_ltp = small_ltpp[:,self.selected_idx]
                filtered_value = np.sum(small_ltp*self.weights[self.selected_idx] , axis=1) \
                            /(np.sum(self.weights[self.selected_idx]))
                small_ltfv.append(filtered_value[:self.ydim*self.k])
                # small_ltsn.append(np.power(10,filtered_value[self.ydim*self.k:self.ydim*self.k+self.sys_pdim]))
                # small_lton.append(np.power(10,filtered_value[self.ydim*self.k+self.sys_pdim:]))
            if self.TIME == 50:
                for taui in range(1):
                    self.long_term_predicted_value.append(small_ltfv[taui]) 
            if self.TIME > 50:
                self.long_term_predicted_value.append(copy.copy(small_ltfv[-1]))
            
            del small_ltpv
            del small_ltpp
            del small_ltp
            del small_ltfv
            # del small_ltsn
            # del small_lton
            gc.collect()
        self.TIME += 1

    def FGHset(self, k, vn_y, n_h_parameters):
        """状態空間表現の行列設定
        vn_y:入力ベクトルの次元
        n_h_parameters:ハイパーパラメタ数
        k:階差
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

# def get_slected_particles_idx(cum,p):
#     """multiprocessing function"""
#     try:
#         return np.where(cum>=p)[0][0]
#     except Exception== e:
#         import sys
#         import traceback
#         sys.stderr.write(traceback.format_exc())    

if __name__=='__main__':
    n_particle = 10000
    nu=0.01
    xi=0.35
    pf = ParticleFilter(n_particle, k=1, ydim=1, sys_pdim=1, ob_pdim=1, sh_parameters=[nu, xi])
    pf.init_praticles_distribution(0, 8) # P, r

    data = np.hstack((norm.rvs(0,1,size=20),norm.rvs(10,1,size=60),norm.rvs(-30,0.5,size=20)))
    
    for d in data:
        pf.forward(d)
    print ('log likelihood:' + str(pf.log_likelihood))
    print ('LSM:'+ str(pf.LSM))

    rng = range(data.shape[0])
    rng1 = range(data.shape[0])
    plt.plot(rng,data,label=u"training data")
    plt.plot(rng,pf.predicted_value,label=u"predicted data")
    plt.plot(rng,pf.filtered_value,label=u"filtered data")
    plt.plot(rng1,pf.long_term_predicted_value,label=u"long-term predicted data")
    plt.ylim((-1000,1000))
#    plt.plot(rng,pf.sys_nois,label=u"system noise hyper parameter")
#    plt.plot(rng,pf.ob_nois,label=u"observation noise hyper parameter")
    plt.xlabel('TIME',fontsize=18)
    plt.ylabel('Value',fontsize=18)    
    plt.legend(loc = 'upper left') 
    plt.show()