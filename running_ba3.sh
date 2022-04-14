dataset_name=ba3

for lr in 1e-5 1e-4 1e-3
do
  for l2 in 1e-5 1e-4 1e-3
  do
    for reward_mode in mutual_info # binary corss_entropy
    do
      python final_run.py --dataset_name $dataset_name --lr $lr --l2 $l2 --reward_mode $reward_mode
    done
  done
done