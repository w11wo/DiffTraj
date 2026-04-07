for city in Beijing Porto San_Francisco; do
    python infer.py \
        --model_path models/${city}_steps=500_len=200_0.05_bs=1024/models/*/unet_best.pt \
        --input_dir data \
        --city $city \
        --trajectory_length 200
    
    python scripts/map_match.py \
        --dataset_path data/${city} \
        --roadmap_geo_path data/${city}/roadmap.geo \
        --city $city \
        --label_trajs_path models/${city}_steps=500_len=200_0.05_bs=1024/results/label_trajs.npy \
        --gen_trajs_path models/${city}_steps=500_len=200_0.05_bs=1024/results/gen_trajs.npy
    
    python eval.py \
        --roadmap_geo_path data/${city}/roadmap.geo \
        --city $city \
        --label_trajs_path models/${city}_steps=500_len=200_0.05_bs=1024/results/label_trajs_map_matched.csv \
        --gen_trajs_path models/${city}_steps=500_len=200_0.05_bs=1024/results/gen_trajs_map_matched.csv
done