for i in "dfdc /data-0/facebook-dataset/final/" "celebdf /data-0/celeb_df_2/raw/" "ff++ /data-0/ff++/raw/" "wilddf /data-0/FFIW10K-v1-release/"
do
    set -- $i # convert the "tuple" into the param args $1 $2...
    echo $1 and $2
   python deepfake_detector/dfdetector.py \
   --benchmark True \
   --detection_method "mesonet_dfdc" \
   --data_path $2 \
   --dataset $1 \
   --wandb True \
    --experiment_group "mesonet_dfdc_cross_dataset"
done