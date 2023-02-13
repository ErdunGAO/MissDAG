
python main.py  --seed 0 \
                --n 100 \
                --d 20 \
                --miss_type mcar \
                --miss_percent 0.2 \
                --graph_type ER \
                --degree 4 \
                --equal_variances \
                --MLEScore Sup-G \
                --noise_type gumbel \
                --miss_method_type miss_dag_nongaussian \
                --em_iter 10 \
                --dag_method_type notears_ica_mcem