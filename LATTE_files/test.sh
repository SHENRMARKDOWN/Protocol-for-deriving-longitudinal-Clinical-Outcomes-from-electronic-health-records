#!/bin/bash
# My first script
#SBATCH -c 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 1:00:00                     # Runtime in D-HH:MM format
#SBATCH -p short                   # Partition to run in
#SBATCH --mem=10G                          # Memory total in MB (for all cores)
#SBATCH -o jobs/testmulti_JID%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e jobs/testmulti_JID%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=juehou@hsph.harvard.edu   # Email to which notifications will be sent

module load gcc/6.2.0
module load conda2/4.2.13
#source activate env_incident
source activate /home/jh502/.conda/envs/env_incident

# updated
python3 c_main.py --train_dfname "example_input/train.csv" \
                  --test_dfname "example_input/test.csv" \
                  --ftsname "example_input/w_fts.csv" \
                  --other_ftsname "example_input/x_fts.csv" \
                    --embed_dim 14 \
                    --embed_fname "example_input/embeddings.csv" \
                    --key_code "W1" \
                    --output_directory "example_out/" \
                    --output_fname "synthetic_test" \
                    --epochs 10 \
                    --max_visits 24 \
                    --flag_train_augment 1 \
                    --flag_cross_dataset 0 \
                    --number_labels 300 \
                    --epoch_silver 0 \
                    --layers_incident "120" \
                    --weight_prevalence 0.3 \
                    --weight_unlabel 0.3 \
                    --weight_contrastive  0.0 \
                    --weight_smooth  0.0 \
                    --weight_additional  0.0 \
                    --flag_save_attention 1 \
                    --flag_load_model 0 \
                    --flag_prediction 0 \
                    --flag_relapse 0 \
                    --has_other_fts \
                    --multi_model "adj_logit" # "base_logit", "cum_logit", "adj_logit"

