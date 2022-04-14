from torch.autograd import Variable
from module.attribution_model_zoo.basic_explainer import Explainer

class SAExplainer(Explainer):

    def __init__(self, gnn_model_path):
        super(SAExplainer, self).__init__(gnn_model_path)

    def explain_graph(self, graph,
                      model=None,
                      draw_graph=0,
                      vis_ratio=0.2):

        if model == None:
            model = self.model

        edge_attr = Variable(graph.edge_attr, requires_grad=True)

        pred = model(graph.x,
                    graph.edge_index,
                    edge_attr,
                    graph.batch
                    )
        pred[0, graph.y].backward()
        edge_grads = pow(edge_attr.grad, 2).sum(dim=1).cpu().numpy()
        edge_imp = self.norm_imp(edge_grads)
        self.last_result = (graph, edge_imp)

        if draw_graph:
            self.visualize(graph, edge_imp, 'SA', vis_ratio=vis_ratio)
        return edge_imp