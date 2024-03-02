python main.py --epochs 1000 --dataset Sin --repeat 5 --un True

python mplcon_evaluate.py --train_dataset Sin --test_dataset Sin --epochs 100
python mplcon_evaluate.py --train_dataset Sin --test_dataset StyleGAN --epochs 100
python mplcon_evaluate.py --train_dataset Sin --test_dataset NeuralTextures --epochs 100