# -*- coding: utf-8 -*-
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import io

def networkx_test():
    path_data_dict = {"pageId":"dongtaiye","times":1}
    G = nx.Graph()
    print type(G)
    # 判断是不是图
    if(str(type(G)).__contains__("graph")):
    	print "shi tu"
    else:
    	print "bushi"

    G.add_edge(1, 2, times=1, operation='op1')
    print G.node



    # if(len(G.node(1)) is 0):
    #     G.add_node(1,data=path_data_dict)
    # print G.nodes()
    # print G.node[1]
    exit()

    G.add_edge(1,2,times=1,operation='op1')
    # 获取边的属性
    print G[1][2]['times']
    G[1][2]['times'] = G[1][2]['times']+1
    print G[1][2]['times']
    print G[1][2]['operation']
    G.add_node(3,attr=path_data_dict)
    print G.node[3]
    current_node_id = "com.qzonex.module.login.ui.QZoneLoginActivityview_scroll45c8fd3b81c7c458eafc99cbf4c449a8"
    #G.add_node(current_node_id,data=path_data_dict)
    if(G.node(current_node_id)):
    	print "shikeyide..."
    print "aaaaaaaaaaaaaaaa\n",type(G.node),G.node
    # 判断点是否存在
    if(G.node[3] is None):
    	G.add_node(3,attr=2)
    try:
    	if(G.node['a']):
    		print "not none"
    except:
    	print "is None"
    print G.node[3]
   
    # 判断边是否存在
    # print G.edges[1][2] 
    # if(G.adj[2][3]):
    #     print "这个方法好使"
    nx.draw(G,node_color = 'b',edge_color = 'r',with_labels = True)
    plt.show()
    exit()

if __name__ == '__main__':
    networkx_test()