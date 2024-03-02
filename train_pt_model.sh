python get_pt_model.py --epochs 1000 --dataset Sin

python evaluate.py --train_dataset Sin --test_dataset Sin --epochs 100
python evaluate.py --train_dataset Sin --test_dataset StyleGAN --epochs 100
python evaluate.py --train_dataset Sin --test_dataset NeuralTextures --epochs 100