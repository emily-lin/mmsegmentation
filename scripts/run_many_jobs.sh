set -e
set -x

ROOT_DIR='/export/data/yuhlab1/emily/mmsegmentation/configs/swin/'

array=( '20220417_weights_v1.py' '20220417_weights_v2.py' '20220417_weights_v3.py' '20220417_weights_v4.py' '20220417_weights_v5.py' '20220417_weights_v6.py' '20220417_weights_v7.py' '20220417_weights_v8.py' '20220417_weights_v9.py'  )
for cfg in "${array[@]}"
do
  ./tools/dist_train.sh "${ROOT_DIR}${cfg}" 3 --load-from work_dirs/20220214_b60_i80k/latest.pth
done
