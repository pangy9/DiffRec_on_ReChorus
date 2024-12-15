cuda=1
path=../my_dataset

dataset="MovieLens_clean"
# dataset="AmazonBook_clean"
# dataset="Grocery_and_Gourmet_Food_clean"
lr=5e-05
weight_decay=0.0
batch_size=400
dims="[1000]"
emb_size=10
mean_type="x0"
steps=5
noise_scale=0.005
noise_min=0.001
noise_max=0.01
sampling_steps=0
reweight=1
w_min=0.5
w_max=1.0

CUDA_VISIBLE_DEVICES=${cuda} python diffrec_main.py \
    --path="${path}" \
    --optimizer AdamW \
    --model_name=TDiffRec \
    --emb_size=${emb_size} \
    --lr=${lr} \
    --dataset="${dataset}" \
    --weight_decay=${weight_decay} \
    --batch_size=${batch_size} \
    --eval_batch_size=${batch_size} \
    --dims="${dims}" \
    --mean_type="${mean_type}" \
    --steps=${steps} \
    --noise_scale=${noise_scale} \
    --noise_min=${noise_min} \
    --noise_max=${noise_max} \
    --sampling_steps=${sampling_steps} \
    --reweight=${reweight} \
    --gpu=${cuda} \
    --num_neg 0\
    --w_min ${w_min} \
    --w_max ${w_max} \
    --random_seed 0
    # --topk 10,20,50,100 \
    # --main_metric "Recall@20" \
    # --metric "DIFFREC"

# 参考
dataset="Grocery_and_Gourmet_Food_clean"
lr=1e-04
weight_decay=0.0
batch_size=400
dims="[1000]"
emb_size=10
mean_type="x0"
steps=5
noise_scale=0.005
noise_min=0.001
noise_max=0.01
sampling_steps=0
reweight=1
w_min=0.5
w_max=1.0