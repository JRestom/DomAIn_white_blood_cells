python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=Naive \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=Joint \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=Cumulative \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=EWC \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=LwF \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=SynapticIntelligence \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet

python classification.py \
  --architecture=Resnet \
  --weights \
  --strategy=Replay \
  --n_classes=5 \
  --epochs=1 \
  --save_dir=/home/ivo.navarrete/Desktop/PhD/CV805/project/DomAIn_white_blood_cells/results/resnet