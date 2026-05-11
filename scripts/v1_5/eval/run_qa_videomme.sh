CKPT_NAME="Video-LLaVA-7B"
model_path="LanguageBind/Video-LLaVA-7B"
cache_dir="./cache_dir"
video_dir="eval/GPT_Zero_Shot_QA/VideoMME_Zero_Shot_QA/videos/data"
gt_file="eval/GPT_Zero_Shot_QA/VideoMME_Zero_Shot_QA/videomme/test-00000-of-00001.parquet"
GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"

default_output_dir="${GPT_Zero_Shot_QA}/VideoMME_Zero_Shot_QA/${CKPT_NAME}_N"
output_dir="${OUTPUT_DIR:-${default_output_dir}}"

export DECORD_EOF_RETRY_MAX=40960
export DECORD_NUM_THREADS=1

SUBSET_ARG=""
if [ -n "${SUBSET_CSV}" ]; then
    SUBSET_ARG="--subset_csv ${SUBSET_CSV}"
    echo "[rerun mode] Filtering samples via ${SUBSET_CSV}"
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
durations="${DURATIONS}"
CHUNKS=${#GPULIST[@]}
mkdir -p ${output_dir}
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_mme.py \
        --model_path ${model_path} \
        --cache_dir ${cache_dir} \
        --video_dir ${video_dir} \
        --gt_file ${gt_file} \
        --output_dir ${output_dir} \
        --output_name ${CHUNKS}_${IDX} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --durations ${durations} \
        ${SUBSET_ARG} &
done
wait
output_file=${output_dir}/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
