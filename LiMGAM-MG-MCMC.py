from collections import defaultdict
import math
import numpy as np
from scipy.stats import gennorm
from sklearn.linear_model import LinearRegression
from causallearn.utils.cit import CIT
from causallearn.score.LocalScoreFunction import local_score_BIC
from causallearn.score.LocalScoreFunction import local_score_cv_general
import matplotlib.pyplot as plt


def graphs_dis(graphs):

    unique_graphs = []

    for edges in graphs:
        G = sorted(edges)
        found = False
        for i, (unique_G, count) in enumerate(unique_graphs):
            if G==unique_G:
                unique_graphs[i] = (unique_G, count + 1)
                found = True
                break
        if not found:
            unique_graphs.append((G, 1))

    most_common_graph, most_common_count = max(unique_graphs, key=lambda x: x[1])
    print(f"Most common graph edges: {most_common_graph}")
    print(f"Frequency: {most_common_count}")

    print("---")
    for G, count in unique_graphs:
        print(f"Graph edges: {G}, Count: {count}")


def graphs_sim(base_graph, graphs):

    set1 = set(base_graph)
    simLis = []
    for graph in graphs:
        set2 = set(graph)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union)
        simLis.append(similarity)
    return simLis


def graphs_und_sim(base_graph, graphs):
    set1 = set(tuple(sorted(edge)) for edge in base_graph)
    simLis = []
    for graph in graphs:
        set2 = set(tuple(sorted(edge)) for edge in graph)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union)
        simLis.append(similarity)
    return simLis



def plot_graph_similarities(base_graph, graph_data, name):

    directed_sim = graphs_sim(base_graph, graph_data)
    undirected_sim = graphs_und_sim(base_graph, graph_data)
    
    mean_directed_sim = sum(directed_sim) / len(directed_sim)
    mean_undirected_sim = sum(undirected_sim) / len(undirected_sim)
    
    fig, axs = plt.subplots(2, 2, figsize=(9, 9), dpi=100)
    
    axs[0][0].plot(range(len(directed_sim)), directed_sim, linewidth=0.2)
    axs[0][0].axhline(y=mean_directed_sim, color='r', linestyle='--')
    axs[0][0].set_ylim(-0.1, 1.2)
    axs[0][0].set_title('Directed Graph Similarities')
    
    axs[1][0].hist(directed_sim, bins=100, density=True)
    axs[1][0].set_ylim(0, 120)
    axs[1][0].set_xlim(-0.05, 1.05)
    axs[1][0].set_title('Histogram of Directed Graph Similarities')
    
    axs[0][1].plot(range(len(undirected_sim)), undirected_sim, linewidth=0.2)
    axs[0][1].axhline(y=mean_undirected_sim, color='r', linestyle='--')
    axs[0][1].set_ylim(-0.1, 1.2)
    axs[0][1].set_title('Undirected Graph Similarities')
    
    axs[1][1].hist(undirected_sim, bins=35, density=True)
    axs[1][1].set_ylim(0, 120)
    axs[1][1].set_xlim(-0.05, 1.05)
    axs[1][1].set_title('Histogram of Undirected Graph Similarities')
    
    plt.tight_layout()

    plt.savefig(f"{name}.png", dpi=600)

    plt.show()
    
    graphs_dis(graph_data)

    print(f'Mean directed similarity: {mean_directed_sim}')
    print(f'Mean undirected similarity: {mean_undirected_sim}')


###


class R_MinimalIMAP_MCMC():

    def __init__(self, data, T, gamma):
        self.data = data
        self.T = T
        self.gamma = gamma


    @staticmethod
    def prior(edges: list[tuple], gamma, prior_info=None): # 注意不是pdf，只是正比

        n = len(edges)
        if prior_info:
            return prior_info * math.exp(-gamma * n)
        else:
            return math.exp(-gamma * n)

    @staticmethod
    def loglikelihood(data, i, pas): # GS_Kunzhang

        score = local_score_cv_general(data, i, pas, {"kfold":1, "lambda":0})
        return score
        """
        cov = np.cov(data.T)
        n = data.shape[0]

        if len(pas) == 0:
            return n * np.log(cov[i, i])

        yX = np.mat(cov[np.ix_([i], pas)])
        XX = np.mat(cov[np.ix_(pas, pas)])
        H = np.log(cov[i, i] - yX * np.linalg.inv(XX) * yX.T)

        return n * H # + np.log(n) * len(pas) * 1
        """

    @staticmethod
    def upd_order(order, s=0.3):

        U = np.random.uniform(0,1)
        if U<s:
            return order
        else:
            """
            i = np.random.randint(0,len(order))
            j = np.random.randint(0,len(order))
            order[i], order[j] = order[j], order[i]
            return order
            """
            """
            np_array = np.array(order)
            np.random.shuffle(np_array)
            return np_array.tolist()
            """
            i = np.random.randint(1,len(order))
            order[i], order[i-1] = order[i-1], order[i]
            return order

    @staticmethod
    def ciT(data, x_index, y_index, z_index):
        kci_obj = CIT(data, "fisherz")#CIT(data, "kci", approx=True, est_width='median')
        pValue = kci_obj(x_index, y_index, z_index)
        if pValue < 0.05:
            #print("不独立")
            return 1
        else:
            #print("独立")
            return 0


    def minimalIMAP(self, order):
        edges = []
        for oder, ch in enumerate(order):
            for pa in order[0:oder]:
                z = order[0:oder].copy()
                z.remove(pa)
                if self.ciT(self.data, ch, pa, z):
                    edges.append((pa, ch))
        return edges


    def graphLogLikelihood(self, edges):

        L = 0
        parents = defaultdict(list)
        for edge in edges:
            parent, child = edge
            parents[child].append(parent)
        for ch, pas in parents.items():
            score = self.loglikelihood(self.data, ch, pas)
            L += - score.item()
        return L


    def minimalIMAP_MCMC(self, order=None): #prior_info?

        if order == None:
            order = [i for i in range(self.data.shape[1])]
        T = self.T
        edges = self.minimalIMAP(order)
        edgess = [edges]
        while len(edgess) < T:
            new_order = self.upd_order(order)

            ###
            new_edges = self.minimalIMAP(new_order)
            ###
            """
            pi_1 = self.graphLogLikelihood(edges) * self.prior(edges, self.gamma)
            pi = self.graphLogLikelihood(new_edges) * self.prior(new_edges, self.gamma)
            cri = math.exp(pi - pi_1)
            #print(cri)
            U = np.random.uniform(0,1)

            if U < cri:
                order = new_order
                edges = new_edges
                edgess.append(edges)
            """
            order = new_order
            edges = new_edges
            edgess.append(edges)
        return edgess


###


class MinimalIMAP_MCMC():

    def __init__(self, data, T, gamma):
        self.data = data
        self.T = T
        self.gamma = gamma


    @staticmethod
    def prior(edges: list[tuple], gamma, prior_info=None): # 注意不是pdf，只是正比

        n = len(edges)
        if prior_info:
            return prior_info * math.exp(-gamma * n)
        else:
            return math.exp(-gamma * n)

    @staticmethod
    def loglikelihood(data, i, pas): # GS_Kunzhang

        score = local_score_cv_general(data, i, pas, {"kfold":1, "lambda":0})
        return score
        """
        cov = np.cov(data.T)
        n = data.shape[0]

        if len(pas) == 0:
            return n * np.log(cov[i, i])

        yX = np.mat(cov[np.ix_([i], pas)])
        XX = np.mat(cov[np.ix_(pas, pas)])
        H = np.log(cov[i, i] - yX * np.linalg.inv(XX) * yX.T)

        return n * H # + np.log(n) * len(pas) * 1
        """

    @staticmethod
    def upd_order(order, s=0.3):

        U = np.random.uniform(0,1)
        if U<s:
            return order
        else:
            """
            i = np.random.randint(0,len(order))
            j = np.random.randint(0,len(order))
            order[i], order[j] = order[j], order[i]
            return order
            """
            """
            np_array = np.array(order)
            np.random.shuffle(np_array)
            return np_array.tolist()
            """
            i = np.random.randint(1,len(order))
            order[i], order[i-1] = order[i-1], order[i]
            return order

    @staticmethod
    def ciT(data, x_index, y_index, z_index):
        kci_obj = CIT(data, "fisherz")#CIT(data, "kci", approx=True, est_width='median')
        pValue = kci_obj(x_index, y_index, z_index)
        if pValue < 0.05:
            #print("不独立")
            return 1
        else:
            #print("独立")
            return 0


    def minimalIMAP(self, order):
        edges = []
        for oder, ch in enumerate(order):
            for pa in order[0:oder]:
                z = order[0:oder].copy()
                z.remove(pa)
                if self.ciT(self.data, ch, pa, z):
                    edges.append((pa, ch))
        return edges


    def graphLogLikelihood(self, edges):

        L = 0
        parents = defaultdict(list)
        for edge in edges:
            parent, child = edge
            parents[child].append(parent)
        for ch, pas in parents.items():
            score = self.loglikelihood(self.data, ch, pas)
            L += score.item()
        return L


    def minimalIMAP_MCMC(self, order=None): #prior_info?

        if order == None:
            order = [i for i in range(self.data.shape[1])]
        T = self.T
        edges = self.minimalIMAP(order)
        edgess = [edges]
        while len(edgess) < T:
            new_order = self.upd_order(order)

            ###
            new_edges = self.minimalIMAP(new_order)
            ###

            pi_1 = self.graphLogLikelihood(edges) * self.prior(edges, self.gamma)
            pi = self.graphLogLikelihood(new_edges) * self.prior(new_edges, self.gamma)
            cri = math.exp(pi - pi_1)
            #print(cri)
            U = np.random.uniform(0,1)

            if U < cri:
                order = new_order
                edges = new_edges
                edgess.append(edges)
        
        return edgess
    

###


class MG_MinimalIMAP_MCMC():

    def __init__(self, data, T, gamma):
        self.data = data
        self.T = T
        self.gamma = gamma
        self.order1, self.order2 = self.separMG(self.data)


    def separMG(self, data):

        def indtest(x1,x2):
            data = np.column_stack((x1, x2))
            kci_obj = CIT(data, "kci")
            pValue = kci_obj(0, 1)
            if pValue < 0.01:
                #print("不独立")
                return 1
            else:
                #print("独立")
                return 0


        def compute_residuals(X, y):
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            return residuals
        rs, cs = data.shape

        regind = []
        for i in range(cs):
            y = data[:,i]
            rs = []
            for j in range(cs):
                if i==j:
                    rs.append(0)
                else:
                    x = data[:,j]
                    r = compute_residuals(x.reshape(-1, 1),y)
                    rs.append(indtest(x,r))
            regind.append(rs)

        for row in regind:
            print(row)

        order1 = []
        order2 = []
        for indx, col in enumerate(np.array(regind).T):
            if col.any()==0:
                order1.append(indx)
            else:
                order2.append(indx)
        return order1, order2


    def prior(self, edges: list[tuple], order, gamma, prior_info=None): # 注意不是pdf，只是正比

        n = len(edges)
        if order[0] in self.order1 and order[-1] in self.order2:
            return 1000 * math.exp(-gamma * n)
        else:
            return math.exp(-gamma * n)


    @staticmethod
    def loglikelihood(data, i, pas): # GS_Kunzhang

        score = local_score_cv_general(data, i, pas, {"kfold":1, "lambda":0})
        return score
        """
        cov = np.cov(data.T)
        n = data.shape[0]

        if len(pas) == 0:
            return n * np.log(cov[i, i])

        yX = np.mat(cov[np.ix_([i], pas)])
        XX = np.mat(cov[np.ix_(pas, pas)])
        H = np.log(cov[i, i] - yX * np.linalg.inv(XX) * yX.T)

        return n * H # + np.log(n) * len(pas) * 1
        """


    def upd_order(self, order, s=0.3):

        U = np.random.uniform(0,1)
        if U<s:
            return order
        else:
            """
            i = np.random.randint(0,len(order))
            j = np.random.randint(0,len(order))
            order[i], order[j] = order[j], order[i]
            return order
            """
            """
            np_array = np.array(order)
            np.random.shuffle(np_array)
            return np_array.tolist()
            """
            order1 = self.order1
            if len(order1)>1:
                i = np.random.randint(1,len(order1))
                order1[i], order1[i-1] = order1[i-1], order1[i]
            order2 = self.order2
            if len(order2)>1 and U>0.7:
                i = np.random.randint(1,len(order2))
                order2[i], order2[i-1] = order2[i-1], order2[i]
            return order1+order2


    @staticmethod
    def ciT(data, x_index, y_index, z_index):
        kci_obj = CIT(data, "fisherz")#CIT(data, "kci", approx=True, est_width='median')
        pValue = kci_obj(x_index, y_index, z_index)
        if pValue < 0.05:
            #print("不独立")
            return 1
        else:
            #print("独立")
            return 0


    def minimalIMAP(self, order):
        edges = []
        for oder, ch in enumerate(order):
            for pa in order[0:oder]:
                z = order[0:oder].copy()
                z.remove(pa)
                if self.ciT(self.data, ch, pa, z):
                    edges.append((pa, ch))
        return edges


    def graphLogLikelihood(self, edges):

        L = 0
        parents = defaultdict(list)
        for edge in edges:
            parent, child = edge
            parents[child].append(parent)
        for ch, pas in parents.items():
            score = self.loglikelihood(self.data, ch, pas)
            L += score.item()
        return L


    def minimalIMAP_MCMC(self, order=None): #prior_info?

        if order == None:
            order = self.order1+self.order2
        T = self.T
        edges = self.minimalIMAP(order)
        edgess = [edges]
        while len(edgess) < T:
            new_order = self.upd_order(order)

            ###
            new_edges = self.minimalIMAP(new_order)
            ###

            pi_1 = self.graphLogLikelihood(edges) * self.prior(edges, order, self.gamma)
            pi = self.graphLogLikelihood(new_edges) * self.prior(new_edges, new_order, self.gamma)
            cri = math.exp(pi - pi_1)
            #print(cri, new_order)
            U = np.random.uniform(0,1)

            if U < cri:
                order = new_order
                edges = new_edges
                edgess.append(edges)
        
        return edgess