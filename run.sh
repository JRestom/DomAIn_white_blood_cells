# Run the main.py file with the following arguments

# Testing argparser
# python main.py \
#   --architecture=Resnet \
#   --weights \
#   --strategy=Naive \
#   --n_classes=5
#   --epochs=1

# Run main.py for Biomed_clip
python main.py \
  --architecture=Biomed_clip \
  --weights \
  --strategy=Naive \
  --n_classes=5 \
  --epochs=5

  python main.py \
  --architecture=Biomed_clip \
  --weights \
  --strategy=Joint \
  --n_classes=5 \
  --epochs=5

  python main.py \
  --architecture=Biomed_clip \
  --weights \
  --strategy=EWC \
  --n_classes=5 \
  --epochs=5

  python main.py \
  --architecture=Biomed_clip \
  --weights \
  --strategy=Replay \
  --n_classes=5 \
  --epochs=5