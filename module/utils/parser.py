import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforced Screener")

    # # ===== dataset ===== #
    # parser.add_argument("--dataset", nargs="?", default="mutag", help="Choose a dataset:[last-fm,amazon-book,yelp2018]")
    #
    # # ===== train ===== #
    # parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--dim', type=int, default=64, help='embedding size')
    # parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    # parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    # parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    # parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    # parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    # parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    # parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    # parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    # parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    # parser.add_argument("--gpu_id", type=int, default=6, help="gpu id")
    # parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    # parser.add_argument('--test_flag', nargs='?', default='part',
    #                     help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    # parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    #
    # # ===== relation context ===== #
    # parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--dataset_name", type=str, default="vg", help="sigmoid, softmax")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--reward_mode", type=str, default="binary", help="cross_entropy, binary")

    # parser.add_argument("--edge_scoring_type", type=str, default="sigmoid", help="sigmoid, softmax")
    # parser.add_argument("--reward_type", type=str, default="binary", help="cross_entropy, binary")
    # parser.add_argument("--reward_discount_type", type=str, default="ascending", help="ascending, null, descending")
    # parser.add_argument("--optimize_scope", type=str, default="all", help="all, part")
    # parser.add_argument("--multiple_explainers", type=bool, default=True, help="use multiple or single explainer(s)")

    return parser.parse_args()