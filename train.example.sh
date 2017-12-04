
IMGMODEL="vgg19"
EBT="glove"
EBD=300

source activate vqa_env
python train.py -img $IMGMODEL -ebt $EBT -ebd $EBD
source deactivate