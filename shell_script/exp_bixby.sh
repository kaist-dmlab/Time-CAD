for seed in 0
do
    for model in MS-RNN CNN GRU Bi-GRU LSTM
    do
        for temporal in 0
        do
            for decomposition in 0 1
            do
                for segmentation in 0 1
                do
                    for dataset in bixby
                    do
                        for gpu_id in 5
                        do

                            python3 main.py\
                            --seed $seed\
                            --model $model\
                            --temporal $temporal\
                            --decomposition $decomposition\
                            --segmentation $segmentation\
                            --dataset $dataset\
                            --gpu_id $gpu_id\
                            | tee -a ./results/${dataset}_round${seed}.out
                            
                        done
                    done
                done
            done
        done
    done
done
