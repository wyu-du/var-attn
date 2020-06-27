TEXT=data/quora
DATA=data/quora/quora_30
DATATEST=data/quora/quora_30_test

preprocess_bpe(){
    # Preprocesses the data in data/iwslt14-de-en
    # Since we are using BPE, we do not force any unks.
    mkdir -p data/iwslt
    python preprocess.py \
        -train_src ${TEXT}/train.txt.src -train_tgt ${TEXT}/train.txt.tgt.tagged \
        -valid_src ${TEXT}/val.txt.src -valid_tgt ${TEXT}/val.txt.tgt.tagged \
        -src_vocab_size 20000 -tgt_vocab_size 20000 \
        -src_words_min_frequency 0 -tgt_words_min_frequency 0 \
        -src_seq_length 30 -tgt_seq_length 30 \
        -save_data $DATA

    # Get the test data for evaluation
    python preprocess.py \
        -train_src ${TEXT}/train.txt.src -train_tgt ${TEXT}/train.txt.tgt.tagged \
        -valid_src ${TEXT}/test.txt.src -valid_tgt ${TEXT}/test.txt.tgt.tagged \
        -src_vocab_size 20000 -tgt_vocab_size 20000 \
        -src_words_min_frequency 0 -tgt_words_min_frequency 0 \
        -src_seq_length 30 -tgt_seq_length 30 \
        -leave_valid \
        -save_data $DATATEST
}

train_cat_sample_b6() {
    gpuid=0
    seed=3435
    name=model_cat_sample_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode sample \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_cat_gumbel_b6() {
    gpuid=0
    seed=3435
    name=model_cat_gumbel_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode gumbel \
        -batch_size 6 \
        -temperature 0.1 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_cat_wsram_b6() {
    gpuid=0
    seed=3435
    name=model_cat_wsram_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode wsram \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 6 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 5 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_cat_enum_b6() {
    gpuid=0
    seed=3435
    name=model_cat_enum_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode enum \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_exact_b6() {
    gpuid=0
    seed=3435
    name=model_exact_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode exact \
        -use_generative_model 1 \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_soft_b6() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b6
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

eval_cat() {
    model=$1
    python train.py \
        -data $DATATEST \
        -eval_with $model \
        -save_model none -gpuid 0 -seed 131 -encoder_type brnn -batch_size 4 \
        -accum_count 1 -valid_batch_size 1 -epochs 30 -inference_network_type bigbrnn \
        -p_dist_type categorical -q_dist_type categorical -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 -n_samples 1 -mode sample \
        -eval_only 1
}

gen_cat() {
    model=$1
    python translate.py \
        -src ${TEXT}/test.txt.src \
        -beam_size 10 \
        -batch_size 2 \
        -length_penalty wu \
        -alpha 1 \
        -eos_norm 3 \
        -gpu 0 \
        -output $model.out \
        -output2 $model.tgt \
        -model $model
}

gen_cat_k() {
    model=$1
    for k in 1 2 3 4 5; do
        python translate.py \
            -src data/iwslt14-de-en/test.de.bpe \
            -beam_size 10 \
            -k $k \
            -batch_size 2 \
            -length_penalty wu \
            -alpha 1 \
            -eos_norm 3 \
            -gpu 0 \
            -output $model.$k.out \
            -model $model
    done
}

