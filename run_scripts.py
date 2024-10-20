import os
import copy
import numpy as np
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description='test for diffusion model')
parser.add_argument('--dataset',type=str,default="cora",choices=['cora', 'citeseer', 'polblogs', 'pubmed'], help='dataset name')
parser.add_argument('--att_type',type=str,default="meta",choices=['meta', 'nettack', 'random','noattk'], help='attack type')
parser.add_argument('--att_rate',type=str,default="0.05",choices=['0.05', '0.1', '0.15', '0.2', '0.25','1.0', '2.0', '3.0','4.0', '5.0', '0.0'], help='attack rate')
parser.add_argument('--edgeloss_type',type=str,default="featmap",choices=['feat', 'map', 'featmap'], help='edge type')
parser.add_argument('--gpu_id',type=str, default="0", help='gpu id')
args = parser.parse_args()
emb_dir = "/home/local/ASUAD/changyu2/generate_robust_graph_embedding"
model_dir = "/data-drive/backup/changyu/expe"
attk_embd_dir = f"{emb_dir}/gcl_embeddings/{args.dataset}/{args.att_type}_{args.att_rate}_all_embs.npy"
attk_label_dir = f"{emb_dir}/gcl_embeddings/{args.dataset}/{args.att_type}_{args.att_rate}_all_ps_labels.npy"
attk_adj_dir = f"{model_dir}/graphattk/{args.dataset}_{args.att_type}_adj_{args.att_rate}.npz"
vae_dir_1 = f"{model_dir}/gge/robustgraphvae_{args.dataset}_{args.att_type}_{args.att_rate}_only_feat_lr2.4"
vae_dir_2 = f"{model_dir}/gge/robustgraphvae_{args.dataset}_{args.att_type}_{args.att_rate}_freeze_enc_feat_map_lr2.4"

if args.edgeloss_type == 'feat':
    factor_edgemap = '0'
    factor_edgefeat = '1'   
elif args.edgeloss_type == 'map' :
    factor_edgemap = '1'
    factor_edgefeat = '0'
elif args.edgeloss_type == 'featmap':
    factor_edgemap = '1'
    factor_edgefeat = '1'
else:
    raise ValueError('edgeloss_type should be feat, map or featmap')

if args.dataset == 'pubmed':
    feat_emb_dim = 256
else:
    feat_emb_dim = 512

vae_dir_3 = f"{model_dir}/gge/robustgraphvae_{args.dataset}_{args.att_type}_{args.att_rate}_freeze_enc_feat_map_edge{args.edgeloss_type}_lr1.5"
diffusion_dir = f"{model_dir}/gge/unet_1d_{args.dataset}_{args.att_type}_{args.att_rate}_64_robustgvae_encode_all_norm_ema"


labels = np.load(attk_label_dir)
datalen = labels.shape[0]
clsnum = np.unique(labels).shape[0]
genbatch = (3000//clsnum) * clsnum
genum = (datalen//genbatch + 1 ) *  genbatch * 3
vae_encode_data = f"{vae_dir_3}/{args.dataset}_latents_{datalen}_robustgvae_250_64_encode.npy"
diffusion_sample_data = f"{diffusion_dir}/samples_{genum}_diffusion_3000_1.8.npy"
diffusion_sample_label = f"{diffusion_dir}/labels_{genum}_diffusion_3000_1.8.npy"

#result dirs
vae_decode_feature = f"{vae_dir_3}/{args.dataset}_latents_{datalen}_robustgvae_250_64_decode_feat.npy"
vae_decode_map = f"{vae_dir_3}/{args.dataset}_latents_{datalen}_robustgvae_250_64_decode_map.npy"

python_path = "/home/local/ASUAD/changyu2/miniconda3/envs/ldm/bin/python"

def command(args):

    script_vae_1 = f'CUDA_VISIBLE_DEVICES={args.gpu_id}  {python_path}  {emb_dir}/train_robust_graph_vae.py --moddir {vae_dir_1} --samdir {vae_dir_1} --datadir {attk_embd_dir} --labeldir {attk_label_dir} --adjdir {attk_adj_dir} --norm 1 --lr 2e-4 --coef_recon 1 --coef_map 0 --factor 0  --factor_edgemap 0 --interval 2000 --intervalplot 10 --epoch 50000 --freeze 0 --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --feat_emb_dim {feat_emb_dim}'

    script_vae_2 = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {emb_dir}/train_robust_graph_vae.py --moddir {vae_dir_2} --samdir {vae_dir_2} --datadir {attk_embd_dir} --labeldir {attk_label_dir} --adjdir {attk_adj_dir} --checkpoint_path {vae_dir_1}/ckpt_50000_checkpoint.pt --norm 1 --lr 2e-4 --coef_recon 1 --coef_map 1 --factor 0  --factor_edgemap 0 --interval 2000 --intervalplot 10 --epoch 50000 --freeze 1 --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --feat_emb_dim {feat_emb_dim}'
 
    script_vae_3 = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {emb_dir}/train_robust_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {attk_embd_dir} --labeldir {attk_label_dir} --adjdir {attk_adj_dir} --checkpoint_path {vae_dir_2}/ckpt_50000_checkpoint.pt --norm 1 --lr 1e-5 --coef_recon 1 --coef_map 1 --interval 2 --intervalplot 2 --epoch 300 --freeze 1 --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --factor {factor_edgefeat} --factor_edgemap {factor_edgemap} --feat_emb_dim {feat_emb_dim}'

    script_vae_encode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {emb_dir}/train_robust_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {attk_embd_dir} --labeldir {attk_label_dir} --adjdir {attk_adj_dir} --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --run_type encode --lastepo 250 --feat_emb_dim {feat_emb_dim}'

    script_diffusion = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {emb_dir}/train.py --batchsize {datalen} --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --interval 500 --intervalplot 500 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datatype gclemb --datadir {vae_encode_data} --labeldir {attk_label_dir} --genum {clsnum*80} --genbatch {clsnum*40} --norm 1'

    script_sample = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path} {emb_dir}/sample.py --genum {genum} --genbatch {genbatch}  --modch 64 --moddir {diffusion_dir} --samdir {diffusion_dir} --epoch 3000 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum {clsnum} --datadir {vae_encode_data} --labeldir {attk_label_dir} --norm 1'

    script_vae_decode = f'CUDA_VISIBLE_DEVICES={args.gpu_id} {python_path}  {emb_dir}/train_robust_graph_vae.py --moddir {vae_dir_3} --samdir {vae_dir_3} --datadir {attk_embd_dir} --labeldir {attk_label_dir} --adjdir {attk_adj_dir} --dataset {args.dataset} --neighbor_map_dim {datalen} --batchsize {datalen} --run_type decode --lastepo 250 --datadecode_dir {diffusion_sample_data} --labelsdecode_dir {diffusion_sample_label} --feat_emb_dim {feat_emb_dim}'

    return script_vae_1, script_vae_2, script_vae_3, script_vae_encode, script_diffusion, script_sample, script_vae_decode

def run(args):
    #opt_ = copy.deepcopy(args)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commands = command(args)
    for cmd in commands:
        print(cmd)
        os.system(cmd)
    #output results dir to a file
    expe_setting = f"{args.dataset}, {args.att_type}, {args.att_rate}, {args.edgeloss_type}"
    with open(f"{model_dir}/graphattk/commands.txt", 'a') as command_f:
        command_f.write(timestamp +': \n')
        command_f.write(expe_setting +'\n')
        for cmd in commands:
            command_f.write(cmd+'\n')
        command_f.write(f"############################################################################################## \n")    

    with open(f"{model_dir}/graphattk/results.txt", 'a') as result_f:
        result_f.write(timestamp +'\n')
        result_f.write(expe_setting +': \n')
        result_f.write(f"attacked features: \n")
        result_f.write(f"{attk_embd_dir} \n")
        result_f.write(f"ground truth labels stored in : \n")
        if args.att_type == 'pgd':
            result_f.write(f"{model_dir}/graphattk/pgd_{args.dataset}.npz \n")
        else:
            result_f.write(f"{model_dir}/graphattk/lcc_{args.dataset}.npz \n")
        result_f.write(f"attacked adj: \n")
        result_f.write(f"{attk_adj_dir} \n")
        result_f.write(f"split nodes file: \n")
        result_f.write(f"{model_dir}/graphattk/{args.dataset}_nettacked_nodes.json \n")
        result_f.write(f"---------------------------------------------------------------- \n")
        result_f.write(f"----Synthetic emd, labels, map --------------------------------- \n")
        result_f.write(f"{vae_decode_feature} \n")
        result_f.write(f"{diffusion_sample_label} \n")
        result_f.write(f"{vae_decode_map} \n")

        result_f.write(f"############################################################################################## \n")





    


if __name__ == '__main__':
    
    run(args)