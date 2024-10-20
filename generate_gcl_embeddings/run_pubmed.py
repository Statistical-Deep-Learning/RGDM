#pubmed and polblogs use this script

import os
import copy
import argparse

opt = dict()

opt['device'] = 'cuda:0'
opt['seed'] = '2023'
opt['data'] = 'pubmed'
opt['epochs'] = 600
opt['lr'] = 1e-3
opt['input_dim'] = 500
opt['momentum'] = 0.9
opt['alpha'] = 0.05
opt['beta'] = 0.7
opt['drop_edge'] = 0.4
opt['drop_feat1'] = 0.2
opt['drop_feat2'] = 0.2
opt['gnn_dim'] = 256

parser = argparse.ArgumentParser(description='Generate GCL embeddings')
parser.add_argument('--dataset',type=str,default="cora",choices=['cora', 'citeseer', 'polblogs', 'pubmed'], help='dataset name')
parser.add_argument('--att_type',type=str,default="meta",choices=['meta', 'nettack', 'random','noattk', 'pgd'], help='attack type')
parser.add_argument('--att_rate',type=str,default="0.05",choices=['0.05', '0.1', '0.15', '0.2', '0.25','1.0', '2.0', '3.0','4.0', '5.0', '0.0'], help='attack rate')
args = parser.parse_args()
# The directory where the embeddings are saved
opt['embed_path'] = f'/home/local/ASUAD/changyu2/generate_robust_graph_embedding/gcl_embeddings/{args.dataset}/'


if args.att_type == 'pgd':
    opt['data_path'] = f'/data-drive/backup/changyu/expe/graphattk/pgd_{args.dataset}.npz'
else:
    opt['data_path'] = f'/data-drive/backup/changyu/expe/graphattk/lcc_{args.dataset}.npz'
if args.att_type == 'pgd':
    opt['split_path'] = f'/data-drive/backup/changyu/expe/graphattk/{args.dataset}_pgd_nodes.json'
else:
    opt['split_path'] = f'/data-drive/backup/changyu/expe/graphattk/{args.dataset}_nettacked_nodes.json'


if args.att_type == 'noattk':
    opt['att_adj_path'] = f'/data-drive/backup/changyu/expe/graphattk/{args.dataset}_{args.att_type}_adj_{args.att_rate}.npz'
else:
    opt['att_adj_path'] = f'/data-drive/backup/changyu/expe/graphattk/{args.dataset}_{args.att_type}_adj_{args.att_rate}.npz'
opt['att_type'] = args.att_type
opt['att_rate'] = args.att_rate



def command(opt):
    script = '/home/local/ASUAD/changyu2/miniconda3/envs/graphattk/bin/python /home/local/ASUAD/changyu2/generate_robust_graph_embedding/generate_gcl_embeddings/train.py'
    for opt, val in opt.items():
        script += ' --' + opt + ' ' + str(val)
    return script


def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(command(opt_))


if __name__ == '__main__':
    run(opt)