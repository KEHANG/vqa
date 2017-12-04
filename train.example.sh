
IMGMODEL="vgg19"
EBT="glove"
EBD=300
BS=50

source activate vqa_env
THEANO_FLAGS=device=cuda python train.py -img $IMGMODEL -ebt $EBT -ebd $EBD -bs $BS
source deactivate