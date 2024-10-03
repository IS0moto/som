#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================
# @file SOM.py
# @author K.ISO
# @brief 自作SOMクラス（潜在空間の次元数は固定で2）
# ==================================================
# ==================================================
# import
# ==================================================
import numpy as np
import math
import random
from sklearn.decomposition import PCA
np.random.seed(seed=0)


class SOM():
    
    def __init__(self, resolution: int, observe_dim: int, initalize = "pca"):
        """イニシャライズ
        Args:
            width (int): SOMの横幅
            depth (int): SOMの奥行
            height (int): 一ノードのベクトル長
            initialize (str): 初期化方法（pca|random|zero）
        """
        self.height = observe_dim
        self.width = resolution
        self.depth = resolution
        self.resolution = resolution
        self.initalize = initalize
        if initalize == "pca":
            pass
            # self.pcaInitialize()
        elif initalize == "random":
            self.randomInitialize()
        else:
            self.zeroInitialize()
        # 初期学習半径
        self.init_radius = resolution / 2
        # 学習半径係数
        self.radius_decay = 0.5
        # 初期学習率（self.timalize_flag がTrueのときのみ使用）
        self.init_learnrate = 0.5
        # 学習率係数
        self.learn_decay = 0.01

    def getSizeSOM(self):
        return(self.width, self.depth, self.height)
    
    def randomInitialize(self):
        """SOMのランダム初期化
        """
        self.som = np.random.uniform(-1, 1, (self.width, self.depth, self.height))
        
    def zeroInitialize(self):
        """SOMのゼロ初期化
        """
        self.som = np.zeros((self.width, self.depth, self.height),dtype = np.float)
                    
    def pcaInitialize(self ,data):
        w_axis = np.arange(self.resolution)
        d_axis = np.arange(self.resolution)
        meshgrid = np.meshgrid(w_axis, d_axis)
        zeta = np.dstack(meshgrid).reshape(-1, 2)
        pca = PCA(n_components = 2)
        pca.fit(data)
        self.som = pca.inverse_transform(np.sqrt(pca.explained_variance_)[None, :] * zeta)
    
    def setRadiusParam(self,radius_rate: float):
        """学習半径のパラメータ設定
        Args:
            radius_rate (float): 学習半径
        """
        self.init_radius = radius_rate
            
    def getSOM(self):
        """SOMのマッピングデータを取得
        Returns:
            _type_: SOMのマッピングデータ
        """
        som_data = self.som
        return som_data

    def detectWinnerNode(self, data_matrix):
        """勝者ベクトルの座標を取得
        Args:
            data_matrix (list): 入力ベクトル(batch, 1, input_dim)
        """
        data_matrix = np.expand_dims(data_matrix, axis=1) # shape: (n_samples, 1. input_dim)
        som_data = self.getSOM()
        som_data = self.som.reshape(1, som_data.shape[0] * som_data.shape[1], -1)
        norm = (data_matrix - som_data) ** 2
        norm = np.sum(norm, axis = 2)
        c = np.argmin(norm, axis = 1)
        w_width = c % self.resolution
        w_depth = c // self.resolution
        return (w_width, w_depth)
     
    def getGauss(self,winner_node, ep: int = 1):
        """勝者ノードを中心としたガウシアンを作成
        Args:
            winner_node (array): 勝者ノードの座標（array() array()）
            ep (int, optional): 更新重み率. Defaults to 1.
        Returns:
            gauss (list): 勝者ノードを中心としたガウシアン(batch, x, y)
        """
        
        w_axis = np.arange(self.width)
        d_axis = np.arange(self.depth)
        ws, ds = np.meshgrid(w_axis, d_axis)
        ws = np.expand_dims(ws, axis=0)
        ds = np.expand_dims(ds, axis=0)
        w_width = winner_node[0].reshape(winner_node[0].shape[0], 1, 1)
        w_depth = winner_node[1].reshape(winner_node[1].shape[0], 1, 1)
        d_w = ws - w_width
        d_d = ds - w_depth
        

        self.radius_rate = self.init_radius * np.exp(-ep * self.radius_decay)
        gauss = np.exp(-0.5*(d_d ** 2 + d_w ** 2) / (self.radius_rate ** 2))
        return gauss
    
    def overWriteNodeBatch(self, input_data: list, gauss: list):
        """SOMの更新(バッチ学習)

        Args:
            input_data (list): 観測データ (batch, input_dim)
            gauss (list): 勝者ノードを中心としたガウシアン (batch, x, y)
        """
        _g = gauss.reshape(gauss.shape[0], -1)
        _sum_g = np.sum(_g, axis = 0).reshape(-1, 1)        
        self.som = (np.dot(_g.T, input_data) / _sum_g ).reshape(self.resolution, self.resolution, -1)
    
    def overWriteNodeOnline(self, input_data: list, gauss: list, ep: int = 1):
        """SOMの更新(オンライン学習)

        Args:
            input_data (list): 観測データ (batch, input_dim)
            gauss (list): 勝者ノードを中心としたガウシアン (batch, x, y)
            ep (int, optional): 更新重み率. Defaults to 1.
        """

        self.learn_rate = self.init_learnrate*np.exp(-ep * self.learn_decay)
        _s = self.getSOM().reshape(self.resolution, self.resolution, -1)
        update_matrix = input_data - _s
        _g = gauss.reshape(self.resolution, self.resolution, -1)
        self.som = _s + update_matrix * _g * self.learn_rate
        
    def fitBatch(data, iteration = 10):
        """_summary_

        Args:
            data (_type_): _description_
            iteration (int, optional): _description_. Defaults to 10.
        """
        if self.initalize == "pca":
            self.pcaInitialize(data)
        for i in range(iteration):
            winner_node = self.detectWinnerNode(data)
            gauss_field = self.getGauss(winner_node, i)
            self.overWriteNodeBatch(data, gauss_field)
            
    def save(self, file_name = "latent"):
        np.save(f'weight/{file_name}', self.som)
    
    def load(self, file_name = "latent"):
        self.som = np.load(f'weight/{file_name}.npy')
    
    
