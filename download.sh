mkdir -p /hy-tmp/esm_fold
ln -s /hy-tmp/esm_fold .
cd esm_fold
wget https://hf-mirror.com/facebook/esmfold_v1/resolve/main/config.json
wget https://hf-mirror.com/facebook/esmfold_v1/resolve/main/special_tokens_map.json
wget https://hf-mirror.com/facebook/esmfold_v1/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/facebook/esmfold_v1/resolve/main/vocab.txt
mkdir -p /hy-tmp/zymCTRL
ln -s /hy-tmp/zymCTRL .
cd ../zymCTRL
https://huggingface.co/AI4PD/ZymCTRL/resolve/main/vocab.json
https://huggingface.co/AI4PD/ZymCTRL/resolve/main/tokenizer.json
https://huggingface.co/AI4PD/ZymCTRL/resolve/main/config.json
mkdir -p /hy-tmp/prot_t5_xl_uniref50
cd ../unikp
ln -s /hy-tmp/prot_t5_xl_uniref50 .
cd prot_t5_xl_uniref50
wget https://hf-mirror.com/Rostlab/prot_t5_xl_uniref50/resolve/main/config.json
wget https://hf-mirror.com/Rostlab/prot_t5_xl_uniref50/resolve/main/special_tokens_map.json
wget https://hf-mirror.com/Rostlab/prot_t5_xl_uniref50/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/Rostlab/prot_t5_xl_uniref50/resolve/main/spiece.model
