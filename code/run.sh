# python code/main.py --gpu "2" --model_name "Electra" --data_name "cail_small_sc2_char" --mode "sc" --batch_size 4
# python code/main.py --gpu "3" --model_name "Electra" --data_name "cail_small_sc2_char" --mode "base" --batch_size 32

# # python code/main.py --gpu "3" --model_name "TextCNN" --data_name "cail_small_sc2" --mode "sc" --batch_size 128
# # python code/main.py --gpu "3" --model_name "TextCNN" --data_name "cail_small_sc2" --mode "base" --batch_size 128

# # python code/main.py --gpu "0" --model_name "LSTM" --data_name "cail_small_sc2" --mode "sc" --batch_size 128
# # python code/main.py --gpu "0" --model_name "LSTM" --data_name "cail_small_sc2" --mode "base" --batch_size 128

# python code/main.py --gpu "3" --model_name "Transformer" --data_name "cail_small_sc2" --mode "sc" --batch_size 16
# python code/main.py --gpu "0" --model_name "Transformer" --data_name "cail_small_sc2" --mode "base" --batch_size 128

# python code/main.py --gpu "1" --model_name "Transformer_nj" --data_name "cail_small_sc2" --mode "sc" --batch_size 8
# python code/main.py --gpu "1" --model_name "Transformer_nj" --data_name "cail_small_sc2" --mode "sc" --batch_size 128

# python code/main.py --gpu "0" --model_name "Transformer" --data_name "cail_small_c20" --mode "sc" --batch_size 10
# python code/main.py --gpu "1" --model_name "Transformer" --data_name "cail_small_sc2" --mode "base" --batch_size 128

# python code/main.py --gpu "1" --model_name "Transformer" --data_name "cail_small_sc2" --load_path "code/logs/Transformer_cail_small_sc2/2022-05-18-09:49:44/best_model.pt" --mode "base" --batch_size 128


python code/main.py --gpu "1" --model_name "TextCNN" --data_name "cail_small_sc2" --mode "sc" --batch_size 128
python code/main.py --gpu "1" --model_name "LSTM" --data_name "cail_small_sc2" --mode "sc" --batch_size 128
python code/main.py --gpu "1" --model_name "Transformer" --data_name "cail_small_sc2" --mode "base" --batch_size 128
python code/main.py --gpu "1" --model_name "Transformer" --data_name "laic_sc_acc50k3c10" --mode "sc" --batch_size 128

python code/main.py --gpu "1" --model_name "Electra" --data_name "cail_small_sc2" --mode "base" --batch_size 16
python code/main.py --gpu "1" --model_name "Electra" --data_name "laic_sc_c10" --mode "base" --batch_size 16


python code/main.py --gpu "2" --model_name "TopJudge" --data_name "cail_small_sc2" --mode "base" --batch_size 16
python code/main.py --gpu "2" --model_name "TopJudge" --data_name "laic_sc_c10" --mode "base" --batch_size 16
python code/main.py --gpu "2" --model_name "LSTM" --data_name "laic_sc_c10" --mode "base" --batch_size 128
