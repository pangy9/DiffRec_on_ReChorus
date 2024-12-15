cuda=0
path=../my_dataset

# 参考
dataset=MovieLens_clean
lr=0.001
weight_decay=0.0
batch_size=400
dims="[200,600]"
emb_size=10
mean_type="x0"
steps=40
noise_scale=0.005
noise_min=0.005
noise_max=0.01
sampling_steps=0
reweight=1

CUDA_VISIBLE_DEVICES=${cuda} python diffrec_main.py \
    --path="${path}" \
    --optimizer AdamW \
    --model_name=DiffRec \
    --emb_size=${emb_size} \
    --eval_batch_size=${batch_size} \
    --lr=${lr} \
    --dataset="${dataset}" \
    --l2=${weight_decay} \
    --batch_size=${batch_size} \
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
    --pin_memory 1\
    --num_workers 5\
    --random_seed 42 \
    # --topk 10,20,50,100 \
    # --main_metric "Recall@20" \
    # --metric "DIFFREC"


#参考
dataset="AmazonBook_clean"
lr=5e-05
weight_decay=0.0
batch_size=400
dims="[1000]"
emb_size=10
mean_type="x0"
steps=5
noise_scale=0.0001
noise_min=0.0005
noise_max=0.005
sampling_steps=0
reweight=1

dataset="Grocery_and_Gourmet_Food_clean"
lr=0.001
weight_decay=0.0
batch_size=400
dims="[200,600]"
emb_size=10
mean_type="x0"
steps=5
noise_scale=0.5
noise_min=0.001
noise_max=0.01
sampling_steps=0
reweight=1