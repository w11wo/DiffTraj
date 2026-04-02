for city in Beijing Porto San_Francisco; do
    python main.py \
        --input_dir data \
        --city $city \
        --trajectory_length 200
done