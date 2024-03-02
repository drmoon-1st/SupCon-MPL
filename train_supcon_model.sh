python get_supcon.py --epochs 1000 --dataset Sin
python get_classifier.py --epochs 1000 --dataset Sin

python con_evaluate.py --train_dataset Sin --test_dataset Sin --epochs 100
python con_evaluate.py --train_dataset Sin --test_dataset StyleGAN --epochs 100
python con_evaluate.py --train_dataset Sin --test_dataset NeuralTextures --epochs 100