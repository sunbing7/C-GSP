python causal_analysis.py --option=causal --data_dir=data/ImageNet1k/ --test_dir ./results/gan_n8/model-inv3-epoch9 --result_dir=./output/ --model_t=vgg19 --layer=43
python causal_analysis.py --option=act --data_dir=data/ImageNet1k/ --test_dir ./results/gan_n8/model-inv3-epoch9 --result_dir=./output/ --model_t=vgg19 --layer=43

python causal_analysis.py --option=plot --result_dir=./output/ --model_t=vgg19