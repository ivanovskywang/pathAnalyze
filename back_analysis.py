#coding: utf-8
import json
import os
import networkx as nx
import matplotlib.pyplot as plt


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

def generate_network_info():
    #生成网络图
    file_name = "D:/workspace/AIMonkey/UIPath123.txt"
    rd = RawData()
    lines = rd.load_data(file_name)
    is_new_node = False
    last_page_id = ""
    edge_attr = []
    G = nx.DiGraph()
    for i in range(len(lines)):
        current_page_id = lines[i]["pageId"]
        if i > 0:
            last_page_id = lines[i-1]["pageId"]
        if i == 0 or last_page_id != "" and current_page_id != last_page_id:
            is_new_node = True
        if is_new_node:
            G.add_node(current_page_id)
            if G.number_of_nodes() > 1:
                G.add_edge(last_page_id,current_page_id ,attr=edge_attr)
        else:
            edge_attr.append(lines[i])
        is_new_node = False
    nx.draw(G,with_labels=True)
    plt.savefig("network_hanpei.png")
    plt.show()



if __name__ == '__main__':
    r = RawData()
    s='{"timestamps":"1525330858","event":"press_power","pageId":"com.qzonex.app.tab.QZoneTabActivity全部动态","image_uuid":"0e7618d6-eb4c-438c-a75f-9a8600217af4","screenState":"screen_on"}'
    z = r.to_dict(s)
    print z['pageId']
    print r.is_inherit(z, z)
    generate_network_info()