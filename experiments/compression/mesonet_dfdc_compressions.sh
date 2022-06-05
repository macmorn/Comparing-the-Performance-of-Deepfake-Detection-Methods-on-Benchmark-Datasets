for i in 26 32 36 38 40 42
do
   python deepfake_detector/dfdetector.py \
   --benchmark True \
   --detection_method "mesonet_dfdc" \
   --data_path "/data-0/facebook-dataset/final/" \
   --dataset "dfdc" \
   --wandb True \
   --compress $i
done