python causal_analysis.py --option=causal --test_type=npy --test_dir=./output --result_dir=./output/ --model_t=vgg19 --model_t_ae=dense201 --layer=43
python causal_analysis.py --option=act --test_type=npy --test_dir=./output --result_dir=./output/ --model_t=vgg19 --model_t_ae=dense201 --layer=43

python causal_analysis.py --option=plot --result_dir=./output/ --model_t=vgg19