from pgmpy.independencies import Independencies
from pgmpy.models import BayesianModel
from sklearn import metrics
from scipy import sparse as sp
import numpy as np
from decimal import Decimal
import  Utils.util as util
import itertools
import operator
from pgmpy.base import UndirectedGraph as ug
import pandas as pd
from networkx.algorithms import tree
class TAN(BayesianModel):
    """
    Class to represent Tree Augment Naive classifiers (TAN)
    Subclass of Bayesian Model.
    """

    def __init__(self, ebunch=None):
        self.class_node = None
        self.children_nodes = set()
        self.node_order = []
        super(TAN, self).__init__(ebunch)

    def fit(self, data, class_node=None, estimator=None,first_node=None):
        if not class_node:
            if not self.class_node:
                raise ValueError("class node must be specified for the model")
            else:
                class_node = self.class_node
        if class_node not in data.columns:
            raise ValueError("class node: {node} is not present in the given data".format(node=class_node))
        attribute_names = list(data.columns.difference([class_node]))
        for node in attribute_names:
            self.add_edge(class_node, node)

        len_att = len(attribute_names)
        # 1 Create condition mutual information table
        cmi_df = pd.DataFrame(np.zeros((len_att, len_att)),
                                    index=attribute_names, columns=attribute_names)

        # Store the values and corresponding edges of all condition mutual information
        con_mi_dic = {}
        # Filling condition mutual information table
        for col in attribute_names:
            for row in attribute_names:
                # print("col,row",col,row)
                if (col != row) & (cmi_df.loc[col, row] == 0):
                    result_mi = util.con_mutual_infor_score(list(data[col]), list(data[row]), list(data[class_node]))
                    cmi_df.loc[col, row] = result_mi
                    cmi_df.loc[row, col] = result_mi
                    con_mi_dic[(col, row)] = result_mi
                    con_mi_dic[(row, col)] = result_mi

        # Establish undirected graph with edges marked as cmi
        ugraph = ug()
        ugraph.add_edges_from(list(con_mi_dic.keys()), weights=list(con_mi_dic.values()))
        # Establish undirected maximum_spanning_tree
        ugraph = tree.maximum_spanning_tree(ugraph)
        # Change undirected graph into directed graph according to the maximum spanning tree
        if not first_node:
            start_number = np.random.randint(low=0, high=len_att)
            first_node = attribute_names[start_number]
        added_node = []
        remain_node = attribute_names
        added_node.append(first_node)
        remain_node.remove(first_node)
        while len(remain_node)>0:
            node = added_node[0]
            del added_node[0]
            if len(list(ugraph.neighbors(node))) > 0:
                remove_edges = []
                for i in ugraph.neighbors(node):
                    self.add_edge(node, i)
                    if not i in (added_node):
                        added_node.append(i)
                    if i in remain_node:
                        remain_node.remove(i)
                    remove_edges.append((node, i))
                ugraph.remove_edges_from(remove_edges)
        super(TAN, self).fit(data, estimator)




