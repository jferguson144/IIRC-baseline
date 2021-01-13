mkdir models

wget http://jamesf-incomplete-qa.s3.amazonaws.com/iirc.tar.gz
tar -xzf iirc.tar.gz
rm iirc.tar.gz
cd iirc

wget http://jamesf-incomplete-qa.s3.amazonaws.com/context_articles.tar.gz
tar -xzf context_articles.tar.gz
rm context_articles.tar.gz

cd ..

cd numnet_plus

mkdir drop_dataset

cd drop_dataset && mkdir roberta.base && cd roberta.base

wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin

wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json

# Add "output_hidden_state" : true to the roberta config
sed 's/\( *\)"vocab_size\": \([0-9]*\)/\1"vocab_size\": \2,\n\1"output_hidden_states": true/' config.json > config2.json
mv config2.json config.json

wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json

wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt
