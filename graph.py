import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import igraph as ig
import numpy as np
import xlsxwriter

def get_node_list():
    node_list_folder = os.walk("./Data")
    node_list_folder = [x[0].split("\\") for x in node_list_folder]
    node_list = [x[1] for x in node_list_folder[1:]]
    return node_list


def get_friends(node):
    f = open("./Data/{0}/All Friends.txt".format(node))
    friends = f.readlines()
    friends = [friend.split("/")[-1][:-1] for friend in friends]
    friends = friends[1:]
    return friends


def gamma(G, F, i, j, eigenvalues=None):
    """
        G: Graph
        F: influence_matrix (IM)
        i to j
        eigenvalues: eigenvalues of matrix
    """
    if not eigenvalues:
        a = nx.adjacency_matrix(G).toarray()
        eigenvalues = np.linalg.eigh(a).eigenvalues
    g = 0
    for k in range(0, len(eigenvalues)):
        if k == j or k == i:
            continue
        g += np.sign(F[k][i] - F[k][j])
    return g


def search_node(G, name):
    try:
        return list(G.nodes()).index(name)
    except:
        return -1


def k_coreness(G, node, k_core=None):
    try:
        if not k_core:
            k_core = nx.core_number(G)
        list_neibor = list(G.neighbors(node))
        return sum([k_core[x] for x in list_neibor])
    except:
        return -1


def corenessplus(G, node):
    try:
        list_neibor = list(G.neighbors(node))
        return sum([k_coreness(x) for x in list_neibor])
    except:
        return -1


def influence_matrix(G):
    adjacency_matrix = nx.adjacency_matrix(G)
    a = adjacency_matrix.toarray()
    eigh = np.linalg.eigh(a).eigenvalues
    n = 0.85 * (1 / max(eigh))
    i = np.identity(len(eigh))
    return np.linalg.inv(i - n * a)


def compute_coreness(G, k_core=None):
    if not k_core:
        k_core = nx.core_number(G)
    d = {}
    for n in G.nodes():
        d[n] = k_coreness(G, n, k_core=k_core)
    return d


def set_attribute(G, coreness=None, k_core=None, degree=None):
    nx.set_node_attributes(G, coreness, 'coreness')
    nx.set_node_attributes(G, k_core, 'k_core')
    nx.set_node_attributes(G, degree, 'degree')


def save_file_driver_node(filename, values):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(1, 3, 30)
    worksheet.write('A1', 'Top 3 node có ảnh hưởng lớn nhất')
    worksheet.write_string('A2', 'Node name')
    worksheet.write('B2', 'K-core')
    worksheet.write('C2', 'K-coreness')
    worksheet.write('D2', 'Degree')

    i = 3
    for value in values:
        worksheet.write('A' + str(i), value[0])
        worksheet.write('B' + str(i), value[1]['k_core'])
        worksheet.write('C' + str(i), value[1]['coreness'])
        worksheet.write('D' + str(i), value[1]['degree'])
        i += 1
    workbook.close()
    print('File saved to file: ' + filename)


def save_file_detail(filename, values):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 50)
    worksheet.set_column(1, 2, 30)
    worksheet.write('A1', 'Phân tích chi tiết tất cả các node')
    worksheet.write_string('A2', 'Node name')
    worksheet.write('B2', 'K-core')
    worksheet.write('C2', 'K-coreness')
    worksheet.write('D2', 'Degree')

    i = 3
    for value in values:
        worksheet.write('A' + str(i), value[0])
        worksheet.write('B' + str(i), value[1]['k_core'])
        worksheet.write('C' + str(i), value[1]['coreness'])
        worksheet.write('D' + str(i), value[1]['degree'])
        i +=    1
    workbook.close()
    print('File saved to file: ' + filename)



G = nx.Graph()
node_list = get_node_list()
G.add_nodes_from(node_list)
for node in node_list:
    friends = get_friends(node)
    for fr in friends:
        if not G.has_node(fr):
            G.add_node(fr)
        G.add_edge(node, fr)
        # G.add_edge(fr, node)
G.remove_node("")

remove = [node for node, degree in dict(G.degree()).items() if degree < 2]
G.remove_nodes_from(remove)
while len(remove) > 0:
    remove = [node for node, degree in dict(G.degree()).items() if degree < 2]
    G.remove_nodes_from(remove)

k_core = nx.core_number(G)
coreness = compute_coreness(G, k_core=k_core)

set_attribute(G, coreness=coreness, k_core=k_core, degree=dict(G.degree))

# Sắp xếp theo thứ tự k_coreness
sorted_coreness = sorted(G.nodes(data=True), key=lambda x:x[1]['coreness'], reverse=True)

# Lưu vào file excel
save_file_driver_node('InfluenceNode.xlsx', sorted_coreness[0:3])
save_file_detail('Detail.xlsx', G.nodes(data=True))

# Dự đoán cạnh tranh giữa node i và j, nếu > 0 thì i thắng và ngược lại
# print(gamma(influence_matrix(G), search_node(G, input('Name/ ID 1: ')), search_node(G, input('Name/ ID 2: '))))

# Vẽ đồ thị, hiển thị và lưu ra file
# nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_size=20, font_size=14, width=0.1)
# plt.show()
# plt.savefig("graph.png", dpi=3000)
