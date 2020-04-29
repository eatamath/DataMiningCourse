import pandas as pd
import numpy as np
from numpy.linalg import *
import scipy as sp
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.neighbors import *
from collections import *
from pyclust import *
from sklearn.manifold import *
from sklearn.cluster import KMeans




class TreeNode:
    # node_type = 'root' ### 节点类型 [root | innode | leaf]
    # att_parent = None ### 父节点
    # att_childs = [] ### 子节点
    # att_freq = 0 ### 频率
    # att_class = '' ### 类别标记
    # att_level = 0 ### 类别层级
    # att_tm = 0 ### 最新时间
    def __init__(self,nodeType='root',nodeClass='',childNodes=[],freq=0,level=0,tm=0,parent=None):
        super().__init__()
        self.att_childs = []
        self.att_freq = freq
        self.att_class = nodeClass
        self.node_type = nodeType
        self.att_level = level
        self.att_tm = tm
        self.att_parent = parent if parent is not None else self
        self.is_speed = False
        return
    
    def out(self):
        o = {
            'nodetype':self.node_type,
            'att_childs':len(self.att_childs),
            'att_freq':self.att_freq,
            'att_class':self.att_class,
            'att_level':self.att_level,
            'att_tm':self.att_tm,
            'parent':self.att_parent.att_class
        }
        print(str(o))
        return
    
    def display(self):
        s = [self]
        while len(s):
            node = s.pop(0)
            s.extend(node.att_childs)
            node.out()
        return
    
    def get_all_childs(self):
        if self.node_type=='root':
            s = [self] ### exclude the root
            res = []
            while len(s):
                node = s.pop(0)
                res.extend(node.att_childs)
            return res
        else:
            return []
        
    def get_all_nodes(self):
        if self.node_type=='root':
            s = [self] ### exclude the root
            res = []
            while len(s):
                node = s.pop(0)
                res.extend(node.att_childs)
                s.extend(node.att_childs)
            return res
        else:
            return []
    
    def get_max_freq(self):
        if self.node_type=='root':
            c = self.att_childs
            if len(c):
                return max(c,key=lambda x:x.att_freq).att_freq
        return 0

    def get_avg_freq(self):
        if self.node_type=='root':
            all_nodes = self.get_all_childs()
            if len(all_nodes):
                return np.mean(list(map(lambda x:x.att_freq,all_nodes)))
        return 0
        
    def get_node_size(self):
        if self.node_type=='root':
            sz = len(self.get_all_nodes())
            return sz
        else:
            return -1
        
    def check_tree(self):
        if self.node_type=='root':
            nodes = self.get_all_nodes()
            classes = [n.att_class for n in nodes]
            counter = Counter(classes)
            r = list(filter(lambda x:counter[x]>1,counter))
            if len(r):
                print(r)
                raise Warning('not valid')
        else:
            return
        
    def copy(self,root1,root2):
        nodes = [x for x in root1.att_childs]
        for nn in nodes:
            newnode = TreeNode(nn.node_type,nn.att_class,[],nn.att_freq,nn.att_level,nn.att_tm,root2)
            root2.att_childs.append(newnode)
            self.copy(nn,newnode)
        return
    
    def speed(self):
        self.is_speed = True
        self.search = {x.att_class:x for x in self.get_all_nodes()}
        return
    
    def find_class(self,sclass):
        if self.is_speed:
            return self.search[sclass]
        else:
            return None
        
        
class ETree:
    def __init__(self):
        self.root = TreeNode(nodeType='root')
        self.nodes = {'':self.root}
        return

    def get_node_by_class(self,eclass=''):
        try:
            return self.nodes[eclass]
        except Exception as e:
            print(e)
            print('eclass not found')
            return None
        
    def add_node_by_parent(self,new_node,parent_node):
        parent_node.att_childs.append(new_node)
        new_node.att_parent = parent_node
        # self.nodes['']
            
    
    

print('CTree.py init')
