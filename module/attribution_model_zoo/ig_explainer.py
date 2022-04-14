import numpy as np
import torch
from torch.autograd import Variable
from module.attribution_model_zoo.basic_explainer import Explainer

class IGExplainer(Explainer):

    def explain_graph(self, graph,
                        model=None,
                        draw_graph=0,
                        baseline=0,
                        steps=20,
                        node_base=None,
                        edge_base=None,
                        vis_ratio=0.2
                        ):

        if model == None:
            model = self.model
        # for IG and DeepLIFT method and evaluation
        self.node_base = torch.zeros_like(graph.x) if node_base == None else node_base
        self.edge_base = torch.zeros_like(graph.edge_attr) if edge_base == None else edge_base

        y = graph.y
        scale = [baseline + (float(i) / steps) * (1 - baseline) for i in range(0, steps + 1)]
        edge_grads = []
        step_len = float(1 - baseline) / steps

        for i in range(len(scale)):
            edge_attr = Variable(scale[i] * graph.edge_attr, requires_grad=True)
            pred = model(graph.x,
                       graph.edge_index,
                       edge_attr,
                       graph.batch)

            pred[0, y].backward()
            score = pow(edge_attr.grad, 2).sum(dim=1).cpu().numpy()
            edge_grads.append(score * step_len)

            model.zero_grad()

        edge_grads = np.array(edge_grads).sum(axis=0)
        edge_imp = self.norm_imp(edge_grads)
        self.last_result = (graph, edge_imp)
        if draw_graph:
            self.visualize(graph, edge_imp, 'IG', vis_ratio=vis_ratio)
        return edge_imp