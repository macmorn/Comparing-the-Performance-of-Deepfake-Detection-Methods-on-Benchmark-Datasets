for i in "dfdc /data-0/facebook-dataset/final/" "celebdf /data-0/celeb_df_2/raw/" "ff++ /data-0/ff++/raw/" "wilddf /data-0/FFIW10K-v1-release/"
do
    set -- $i # convert the "tuple" into the param args $1 $2...
    echo $1 and $2
   for j in 26 32 36 38 40 42
   do
      python deepfake_detector/dfdetector.py \
      --benchmark True \
      --detection_method "efficientnetb7_dfdc" \
      --data_path $2 \
      --dataset $1 \
      --wandb True \
      --compress $j \
      --experiment_group "efficientnetb7_dfdc_cross_compress" 
   done
done
