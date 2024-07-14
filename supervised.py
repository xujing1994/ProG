from prompt_graph.supervised_train import Supervised_Train
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args
from prompt_graph.data import load4node,load4graph
args = get_args()
seed_everything(args.seed)

if __name__ == '__main__':
    print('Dataset: {}, GNN: {}, Seed: {}'.format(args.dataset_name, args.gnn_type, args.seed))
    st = Supervised_Train(graph_list=None, input_dim=None, dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, seed=args.seed)
    st.supervised_learning()

