import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--attack', nargs='?', default='None', help="Specify a attack method")
    parser.add_argument('--dim', type=int, default=8, help='Dim of latent vectors.')
    parser.add_argument('--layers', nargs='?', default='[8,8]', help="Dim of mlp layers.")
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    parser.add_argument('--path', nargs='?', default='/data/qqt/PoisonFedDLRS/Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ML', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cuda' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--items_limit', type=int, default=50, help='Limit of items.')
    parser.add_argument('--clients_limit', type=float, default=0.001, help='Limit of proportion of malicious clients.')
    parser.add_argument('--model_type',type=str,default='NCF',choices=["NCF","SASrec","BERT4rec","SASrec2","Bert4rec2"])
    parser.add_argument('--train_robust', type=bool, default=False, help='Limit of items.')
    parser.add_argument("--agg",type=str,default="common")
    parser.add_argument("--num_attack",type=int,default=2)
    return parser.parse_args()


args = parse_args()
