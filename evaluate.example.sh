
WF='evaluation/vqa-vgg16-glove300d/model_weights_epoch_11.h5'
IMGMODEL="vgg16"
EBT="glove"
EBD=300
BS=10

source activate vqa_env
python evaluate.py -wf $WF -img $IMGMODEL -ebt $EBT -ebd $EBD -bs $BS
source deactivate