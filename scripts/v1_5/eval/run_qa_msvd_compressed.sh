

CKPT_NAME="Video-LLaVA-7B"
model_path="LanguageBind/Video-LLaVA-7B"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/videos"
gt_file_question="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_a.json"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

prune_c="${PRUNE_COUNT}"
merge_c="${MERGE_COUNT}"
output_dir="GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/${CKPT_NAME}_M${merge_c}_P${prune_c}"
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_qa.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
      --sample_ratio ${SAMPLE_RATIO} \
      --merge_count ${merge_c} \
      --prune_count ${prune_c} &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
