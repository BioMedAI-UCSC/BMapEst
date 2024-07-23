ignore="48"
ground_truth=subject${ignore}.npz
output="results/${ground_truth}"
num_epochs=501
save_every=50

# Commands for running the sequence ablation experiment
# T1-single contrast
python3 optimize_maps.py \
    --exp direct_pixel_1_seq_single_contrast \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 0 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# T1-four contrasts
python3 optimize_maps.py \
    --exp direct_pixel_1_seq_four_contrasts \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 1 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# T1-T2 four contrasts
python3 optimize_maps.py \
    --exp direct_pixel_2_seq \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 2 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# Three sequences: T1 + T2 + T2*
python3 optimize_maps.py \
    --exp direct_pixel_3_seq \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 3 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# Four sequences: T1 + T2 + T2* + DIRPrep
python3 optimize_maps.py \
    --exp direct_pixel_4_seq \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 4 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# Five sequences: T1 + T2 + T2* + DIRPrep + FLAIRPrep
python3 optimize_maps.py \
    --exp direct_pixel_5_seq \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 5 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# Six sequences: T1 + T2 + T2* + DIRPrep + FLAIRPrep + DIRPrep, Image Space Loss, Direct Pixel Optimization, All 3 maps
python3 optimize_maps.py \
    --exp direct_pixel_6_seq \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth}

# Kspace-loss
python3 optimize_maps.py \
    --exp direct_pixel_kspace_loss \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --kspace_loss

# 19 coefficient optimization
python3 optimize_maps.py \
    --exp 19_linear_coef_pixelwise \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --linear_comb_optim \
    --pixelwise


# Single coefficient per image
python3 optimize_maps.py \
    --exp 19_linear_coef_single \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --linear_comb_optim

# CSF Only
python3 optimize_maps.py \
    --exp direct_pixel_csf_only \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --optimize_csf

# GM Only
python3 optimize_maps.py \
    --exp direct_pixel_gm_only \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --optimize_gm

# WM Only
python3 optimize_maps.py \
    --exp direct_pixel_wm_only \
    --data data/brainweb_64 \
    -s mc_flash \
    --flash_seq_num 6 \
    --device cpu \
    --epochs ${num_epochs} \
    --save_every ${save_every} \
    --init_avg \
    --losses mse \
    --sigma 0 \
    --output ${output} \
    --ignore_sub ${ignore} \
    --ground_truth ${ground_truth} \
    --optimize_wm