# -*- coding: utf-8 -*-
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import io

PATH_FILE = "./UIPath_139K.txt"
GENERATED_PATH = "./generate_path.txt"
PATH_RESULT_FILE = "./path_analysis_result.txt"

class RawData(object):
    def __init__(self, string = None):
        self.string = string

    def to_dict(self, string):
        #字符串转字典
        return json.loads(string.decode("utf-8"))

    def load_data(self, filename):
        #从原始文件加载数据
        f = open(filename, "r")
        self.lines = f.readlines()
        f.close()
        self.con = []
        for i in self.lines:
            l = self.to_dict(i)
            self.con.append(l)
        return self.con

    def is_inherit(self, last, current):
        #判断某个操作是否需要转化
        self.no_static_event = ["press_power", "rotation", "press_home", "pop_activity"]
        self.time_interval = int(current["timestamps"]) - int(last["timestamps"])
        return self.time_interval < 200 and (current['event'] not in self.no_static_event)

class Path_generate(object):
    def __init__(self):
        self.Graph_for_show = nx.DiGraph()

    def generate_network_info(self):
        #生成网络图
        file_name = PATH_FILE
        rd = RawData()
        lines = rd.load_data(file_name)
        is_new_node = False
        #last_page_id = ""
        last_node_id = ""	#节点ID = pageId + event + viewId
        edge_attr = []		#边的属性
        G = nx.DiGraph()
        for i in range(len(lines)):
            # current_page_id = lines[i]["pageId"]
            if(lines[i].has_key('viewId') is False):
                lines[i]['viewId'] = " "
            current_node_id = lines[i]["pageId"] + lines[i]['event'] + lines[i]['viewId']
            if i > 0:
                # last_page_id = lines[i-1]["pageId"]
                last_node_id = lines[i-1]["pageId"] + lines[i-1]["event"] + lines[i-1]["viewId"]
            if i == 0 or last_node_id != "" and current_node_id != last_node_id:
                is_new_node = True
            if is_new_node:
                G.add_node(current_node_id)
                if G.number_of_nodes() > 1:
                    G.add_edge(last_node_id,current_node_id ,attr=lines[i])
                    #G.add_weighted_edges_from([(last_node_id,current_node_id,1)]) #给边加权
                    print G.get_edge_data(last_node_id,current_node_id)
                    print G.number_of_edges()
            else:
                edge_attr.append(lines[i])
            is_new_node = False
        nx.draw(G,with_labels=True)
        plt.savefig("network.png")
        plt.show()
        # path=nx.all_pairs_shortest_path(G)     #调用多源最短路径算法，计算图G所有节点间的最短路径
        #print path[0][2] #打印0、2之间最短路径
        #print nx.diameter(G) # 返回图G的直径（最长最短路径的长度）

    def arlong_generate_path_info_sim(self):
        # 按照简单逻辑将上报的路径进行拆分
        # 简单逻辑：新界面是之前遇到过的，则将之前的界面到目前界面的路径拆出来
        file_name = PATH_FILE
        path_result = []
        rd = RawData()
        lines = rd.load_data(file_name)
        last_node_id = ""
        start_page_id_list = []
        edge_attr = []
        G = nx.DiGraph()
        for i in range(len(lines)):
            if(lines[i].has_key('viewId') is False):
                lines[i]['viewId'] = " "
            if(lines[i].has_key('pageId') is False):
                lines[i]['pageId'] = " "
            if(lines[i].has_key('event') is False):
                lines[i]['event'] = " "
            current_node_id = lines[i]['pageId'] + lines[i]['event'] + lines[i]['viewId']
            # 判断是否在起点列表中
            if start_page_id_list.count(current_node_id) is 0:
                #不在列表中，加入
                start_page_id_list.append(current_node_id)
                #并且绘制图
                G.add_node(current_node_id)
                if last_node_id is not "":
                    G.add_edge(last_node_id,current_node_id)
                last_node_id = current_node_id
            elif len(start_page_id_list) == 1:
                continue
            else:
                #在，就先加入，再将路径提取
                start_page_id_list.append(current_node_id)
                G.add_node(current_node_id)
                G.add_edge(last_node_id,current_node_id)
                last_node_id = current_node_id
                # 画图
                print G.number_of_edges()
                nx.draw(G,with_labels=True)
                # plt.savefig("network.png")
                #plt.show()
                #将路径加入path_result
                path_result.append(start_page_id_list)
                # 各种清空
                start_page_id_list = []
                last_node_id = ""
                G.clear()
        if(len(start_page_id_list) > 0):
            path_result.append(start_page_id_list)
        write_result_to_file(path_result)

    def write_result_to_file(self, path_result):
        print path_result
        res_f = open(PATH_RESULT_FILE,'w')
        for path in path_result:
            res_f.write(str(path))
            res_f.write('\n')
        if res_f:
            res_f.close()

    def print_detail_of_graph(self, G):
        if(str(type(G)).__contains__("graph")):
            print "*******************************"
            print "Number of Nodes:",G.number_of_nodes()
            print "Number of Edges:",G.number_of_edges()
            # print "List of Nodes:",G.nodes()
            # print "List of Edges:",G.edges()
            #print "",
            print "*******************************"

        else:
            print "这不是个图哦('A')"
            return

    def generate_network_info_with_weight(self):
        """
        # 生成一个带权重的操作路径网络图
        """
        file_name = PATH_FILE
        rd = RawData()
        lines = rd.load_data(file_name)
        print "共加载操作记录**("+str(len(lines))+")**条"
        last_node_id = ""
        last_node_id_for_show = ""
        G = nx.DiGraph()
        current_node_id = ""
        start_point_id = ""
        end_point_id = ""
        for i in range(len(lines)):
            if(lines[i].has_key('viewId') is False):
                lines[i]['viewId'] = " "
            if(lines[i].has_key('pageId') is False):
                lines[i]['pageId'] = " "
            if(lines[i].has_key('event') is False):
                lines[i]['event'] = " "
            # current_node_id = lines[i]['pageId'] + "#" + lines[i]['event'] + "#" + lines[i]['viewId']
            current_node_id = lines[i]['pageId']
            graph_for_show_id = lines[i]['pageId']
            if(len(start_point_id) == 0):
                start_point_id = current_node_id
            # 把上报信息加入到节点的data字段
            try:
                G.node[current_node_id]
            except:
                # print "New Node"
                G.add_node(current_node_id, data=lines[i])

            # 以出现的次数作为权重，加到边上
            try:
                edge_tmp = G.adj[last_node_id][current_node_id]['times'] + 1
                G.add_edge(last_node_id, current_node_id, times=edge_tmp)
            except:
                # print "new edge"
                G.add_edge(last_node_id, current_node_id, times=1)
            # print G.adj[last_node_id][current_node_id]['times']
            try:
                self.Graph_for_show.node[graph_for_show_id]
            except:
                self.Graph_for_show.add_node(graph_for_show_id)
            self.Graph_for_show.add_edge(last_node_id_for_show,graph_for_show_id)
            last_node_id = current_node_id
            last_node_id_for_show = graph_for_show_id
        end_point_id = current_node_id
        # print_detail_of_graph(G)
        # pos 指的是布局 主要有spring_layout , random_layout,circle_layout,shell_layout。
        # node_color指节点颜色，有rbykw ,同理edge_color.
        # with_labels指节点是否显示名字,size表示大小，font_color表示字的颜色。
        nx.draw(#self.Graph_for_show,
                G,
                node_color='b',
                edge_color='r',
                with_labels=True,
                font_size=20,
                node_size=50)
        plt.savefig("network_arlong.png")
        # self.generate_path_from_network(G)
        source_node = self.get_biggest_degree_of_network(G)
        self.print_detail_of_graph(G)
        if(source_node):
            path_result = self.generate_dfs_path_from_graph(G, source_node)
            if(len(path_result)>0):
                self.devide_path_by_circle_from_list(path_result)
        else:
            print "初始节点选取失败"
        plt.show()

    def generate_path_from_network(self, G):
        """
        generate path of operation from network-graph G
        the start point is the node that has most number of edge
        :param G:
        :return:
        """
        return
        d = G.degree()
        print nx.degree(G)

    def get_biggest_degree_of_network(self, G):
        """
        return the id of biggest degree of graph
        :param G:
        :return:
        """
        if (str(type(G)).__contains__("digraph")):
            # degree_list = nx.number_strongly_connected_components(G)
            # print "degree_list:", degree_list
            degree_histogram = nx.degree_histogram(G)
            # print "degree_histogram:",degree_histogram
            # test = G.degree()
            # print "``````````````",type(test)
            # # print self.sort_by_value(G.degree())
            # print test
            # print sorted(G.degree().items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            biggest_dgree = 0
            the_node = None
            for node in G.node():
                if G.degree(node) > biggest_dgree:
                    biggest_dgree = G.degree(node)
                    the_node = node
            print "最大的度为"+str(biggest_dgree)
            # print "该节点为："+str(the_node)
            return the_node

        elif(str(type(G)).__contains__("graph")):
            degree_list = nx.connected_component_subgraphs(G)
            print "degree_list:", degree_list
        else:
            print "Not Graph!"
            return None

    def sort_by_value(d):
        items = d.items()
        backitems = [[v[1], v[0]] for v in items]
        backitems.sort()
        return [backitems[i][1] for i in range(0, len(backitems))]

    def union_graphs(self, G1, G2):
        """
        return a graph union by G1 and G2
        :param G1:
        :param G2:
        :return:
        """
        if (str(type(G1)).__contains__("digraph")):
            if (str(type(G2)).__contains__("digraph")):
                G = union(G1, G2)
                return G
        return None



    def generate_dfs_path_from_graph(self, G,source = 0):
        path_list = []
        if (str(type(G)).__contains__("digraph")):
            # H = nx.Graph(G)
            path_list = list(nx.dfs_edges(G, source))
            print path_list
        else:
            print "这不是个图哦('A')"
        return path_list

    def devide_path_by_circle_from_list(self, path_list):
        # 按照节点出现重复的逻辑将路path_list进行拆分
        # 最终返回的结果为path_result_list
        path_result_list = []
        start_page_id_list = []
        for item in path_list:
            # 判断是否在起点列表中
            if start_page_id_list.count(item) == 0 or len(start_page_id_list) <= 2:
                #不在列表中，加入
                start_page_id_list.append(item)
            else:
                #在，就先加入，再将路径提取
                start_page_id_list.append(item)
                #plt.show()
                #将路径加入path_result_list
                path_result_list.append(start_page_id_list)
                # 各种清空
                start_page_id_list = []
        if(len(start_page_id_list) > 0):
            path_result_list.append(start_page_id_list)
        self.write_result_to_file(path_result_list)

    def get_shrotest_path_between_2_points(self, G, start_point_id, end_point_id):
        """
        return the shortest path
        :param start_point_id:
        :param end_point_id:
        :return:
        """
        try:
            op_list = nx.shortest_path(G, start_point_id, end_point_id)
            print "get shortest path:", op_list
            print "path length is :", len(op_list)
            return op_list
        except nx.NetworkXNoPath:
            print 'No path'
            return None
        # 记录下
        path = nx.all_pairs_shortest_path(G)  # 调用多源最短路径算法，计算图G所有节点间的最短路径
        print path[0][2]  # 输出节点0、2之间的最短路径序列： [0, 1, 2]

    def dfs_edges(G, source=None, depth_limit=None):
        """Iterate over edges in a depth-first-search (DFS).

        Parameters
        ----------
        G : NetworkX graph

        source : node, optional
           Specify starting node for depth-first search and return edges in
           the component reachable from source.

        depth_limit : int, optional (default=len(G))
           Specify the maximum search depth.

        Returns
        -------
        edges: generator
           A generator of edges in the depth-first-search.

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> list(nx.dfs_edges(G, source=0))
        [(0, 1), (1, 2), (2, 3), (3, 4)]
        >>> list(nx.dfs_edges(G, source=0, depth_limit=2))
        [(0, 1), (1, 2)]

        Notes
        -----
        If a source is not specified then a source is chosen arbitrarily and
        repeatedly until all components in the graph are searched.

        The implementation of this function is adapted from David Eppstein's
        depth-first search function in `PADS`_, with modifications
        to allow depth limits based on the Wikipedia article
        "`Depth-limited search`_".

        .. _PADS: http://www.ics.uci.edu/~eppstein/PADS
        .. _Depth-limited search: https://en.wikipedia.org/wiki/Depth-limited_search

        See Also
        --------
        dfs_preorder_nodes
        dfs_postorder_nodes
        dfs_labeled_edges
        """
        if source is None:
            # edges for all components
            nodes = G
        else:
            # edges for components with source
            nodes = [source]
        visited = set()
        if depth_limit is None:
            depth_limit = len(G)
        for start in nodes:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, depth_limit, iter(G[start]))]
            while stack:
                parent, depth_now, children = stack[-1]
                try:
                    child = next(children)
                    if child not in visited:
                        yield parent, child
                        visited.add(child)
                        if depth_now > 1:
                            stack.append((child, depth_now - 1, iter(G[child])))
                except StopIteration:
                    stack.pop()


if __name__ == '__main__':
    # r = RawData()
    # s='{"timestamps":"1525330858","event":"press_power","pageId":"com.qzonex.app.tab.QZoneTabActivity全部动态","image_uuid":"0e7618d6-eb4c-438c-a75f-9a8600217af4","screenState":"screen_on"}'
    # z = r.to_dict(s)
    # print z['pageId']
    # print r.is_inherit(z, z)
    # networkx_test()
    # generate_network_info()
    n_list = [1,2,3,4,2, 3,4,5,6,1,4, 2,3,4,5,6,5 ,7,7,8,1,7]
    pg = Path_generate()
    pg.devide_path_by_circle_from_list(n_list)
   # exit(3)

    pg.generate_network_info_with_weight()
    # arlong_generate_path_info_sim()