source ./tools/env_activate.sh

# Setup environment variables for performance on Xeon
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6.0.32
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export QKV_WT_BLK_16_0=256
export QKV_WT_BLK_16_1=32 #1P: 64
export QKV_WT_BLK_16_2=32
export QKV_WT_BLK_16_3=16
export QKV_WT_BLK_16_4=2

export QKV_WT_BLK_64_0=64
export QKV_WT_BLK_64_1=32 #1P: 64
export QKV_WT_BLK_64_2=32
export QKV_WT_BLK_64_3=64
export QKV_WT_BLK_64_4=2

export FQKV_WT_BLK_16_0=384 #1P: 768
export FQKV_WT_BLK_16_1=64
export FQKV_WT_BLK_16_2=32
export FQKV_WT_BLK_16_3=16
export FQKV_WT_BLK_16_4=2

export FQKV_WT_BLK_64_0=96 #1P: 192
export FQKV_WT_BLK_64_1=64
export FQKV_WT_BLK_64_2=32
export FQKV_WT_BLK_64_3=64
export FQKV_WT_BLK_64_4=2

export IGEMM_WT_BLK_16_0=512 #1P: 1024
export IGEMM_WT_BLK_16_1=64
export IGEMM_WT_BLK_16_2=32
export IGEMM_WT_BLK_16_3=16
export IGEMM_WT_BLK_16_4=2

export IGEMM_WT_BLK_64_0=128 #1P: 256
export IGEMM_WT_BLK_64_1=64
export IGEMM_WT_BLK_64_2=32
export IGEMM_WT_BLK_64_3=64
export IGEMM_WT_BLK_64_4=2

export OGEMM_WT_BLK_16_0=256 #1P: 1024
export OGEMM_WT_BLK_16_1=128 #1P: 64
export OGEMM_WT_BLK_16_2=32
export OGEMM_WT_BLK_16_3=16
export OGEMM_WT_BLK_16_4=2

export OGEMM_WT_BLK_64_0=64 #1P: 256
export OGEMM_WT_BLK_64_1=128 #1P: 64
export OGEMM_WT_BLK_64_2=32
export OGEMM_WT_BLK_64_3=64
export OGEMM_WT_BLK_64_4=2

export PLN_WT_BLK_100_0=504
export PLN_WT_BLK_100_1=32 #1P: 64
export PLN_WT_BLK_100_2=32
export PLN_WT_BLK_100_3=100
export PLN_WT_BLK_100_4=2

source /opt/intel/sep/sep_vars.sh

input_tokens=1024
output_tokens=128
batch_size=1
num_beams=4
model="EleutherAI/gpt-j-6b"
sharded_model_path=/home/adgubrud/2024ww03_llm_ipex_innersource/frameworks.ai.pytorch.ipex-cpu/examples/cpu/inference/python/llm/sharded/gpt-j-6b

n_runs=1
num_iters=3
num_warmup=1

if [[ $num_beams -eq 1 ]]; then
        greedy_or_beam="--greedy"
else
        greedy_or_beam=""
fi

n_accel_list=(2)
core_range_list=("0-31,32-63")
core_desc_list=("64C_SNC2_0-31_32-63")

max_idx=$(( ${#n_accel_list[*]}-1 ))
for cfg_idx in $(seq 0 $max_idx);do

for exec_id in $(seq 1 $n_runs);do

n_accel="${n_accel_list[cfg_idx]}"
core_range="${core_range_list[cfg_idx]}"
core_desc="${core_desc_list[cfg_idx]}"

#sync
#echo 3 | sudo tee /proc/sys/vm/drop_caches

DIR_GPTJ=blocks/adgubrud-emr-a1-350w-$(uname -r)kernel-gptj-PCP-$core_desc/

mkdir -p $DIR_GPTJ

cmd="deepspeed --num_accelerators $n_accel  --bind_core_list $core_range --bind_cores_to_rank  run.py  --benchmark -m $sharded_model_path --dtype bfloat16 --ipex --autotp  --max-new-tokens $output_tokens --num-iter $num_iters --num-warmup $num_warmup --token-latency --input-tokens $input_tokens --batch-size  $batch_size"
cmd="$cmd $greedy_or_beam"
echo "$cmd"

#emon -i events_ag.txt &> $DIR_GPTJ/gptj-deepspeed-$input_tokens-inputtokens-$num_beams-beams-$batch_size-deepspeed-$core_desc.emonsubset_ag.iteration_$exec_id.dat &
#emon -collect-edp &> $DIR_GPTJ/gptj-deepspeed-$input_tokens-inputtokens-$num_beams-beams-$batch_size-deepspeed-$core_desc.emonedp.iteration_$exec_id.dat &

eval $cmd &> $DIR_GPTJ/gptj-deepspeed-$input_tokens-inputtokens-$num_beams-beams-$batch_size-deepspeed-$core_desc.iteration_$exec_id.log

#emon -stop

done
done
