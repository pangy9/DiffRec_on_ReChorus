cuda=3
path=../my_dataset

dataset="MovieLens_clean"
lr1=0.001
lr2=0.0005
wd1=0.0
wd2=0.0
batch_size=400
n_cate=2
in_dims='[300]'
out_dims='[]'
lamda=0.03
mlp_dims='[300]'
emb_size=10
mean_type='x0'
steps=100
noise_scale=0.005
noise_min=0.005
noise_max=0.02
sampling_steps=0
reweight=1

emb_path="${path}/${dataset}/item_emb.npy"
CUDA_VISIBLE_DEVICES=${cuda} python diffrec_main.py \
    --path="${path}" \
    --LDiffRec \
    --lr=0 \
    --model_name=LDiffRec \
    --cuda \
    --dataset=${dataset} \
    --data_path=${dataset}/ \
    --emb_path=${emb_path} \
    --lr1=${lr1} \
    --lr2=${lr2} \
    --wd1=${wd1} \
    --wd2=${wd2} \
    --batch_size=${batch_size} \
    --eval_batch_size=${batch_size} \
    --n_cate=${n_cate} \
    --in_dims=${in_dims} \
    --out_dims=${out_dims} \
    --lamda=${lamda} \
    --mlp_dims=${mlp_dims} \
    --emb_size=${emb_size} \
    --mean_type=${mean_type} \
    --steps=${steps} \
    --noise_scale=${noise_scale} \
    --noise_min=${noise_min} \
    --noise_max=${noise_max} \
    --sampling_steps=${sampling_steps} \
    --reweight=${reweight} \
    --log_file=${log_file} \
    --gpu=${cuda} \
    --num_neg 0\
    --pin_memory 1\
    --num_workers 5\
    --random_seed 42 \
    # --topk 10,20,50,100 \
    # --main_metric "Recall@20" \
    # --metric "DIFFREC"



#参考
dataset="AmazonBook_clean"
lr1=0.0005
lr2=0.0001
wd1=0.0
wd2=0.0
batch_size=400
n_cate=2
in_dims='[300]'
out_dims='[]'
lamda=0.03
mlp_dims='[300]'
emb_size=10
mean_type='x0'
steps=5
noise_scale=0.01
noise_min=0.005
noise_max=0.01
sampling_steps=0
reweight=0

dataset="Grocery_and_Gourmet_Food_clean"
lr1=0.001
lr2=0.0005
wd1=0.0
wd2=0.0
batch_size=400
n_cate=2
in_dims='[300]'
out_dims='[]'
lamda=0.03
mlp_dims='[300]'
emb_size=10
mean_type='x0'
steps=40
noise_scale=0.005
noise_min=0.005
noise_max=0.02
sampling_steps=0
reweight=1
