#!/bin/bash

usage() {
    cat << EOF
Usage: All options are needed!
    -p: model path
    -d: device_id (e.g., 0 or 0,1 or 2,3 or 0,1,2,3)
    -t: tp (e.g., 1, 2, 4...)
    -m: model_name (single or comma-separated)
    -b: backend (e.g., lmdeploy/vllm/tensorrt-llm)
    -s: service port (default: 8080)
    -n: parallel num (default: 50)
    -r: run locally with two specified APIS (skip starting model server)
    -a: custom API endpoints (comma-separated, e.g., http://localhost:8080,http://localhost:8888)
EOF
}

# Default values
service_port=8080
backend="lmdeploy"
tp=1
parallel=50
run_locally=false
custom_script="gen_pairwise_result.py"
custom_api=""
model_names=()

# Parse command-line arguments
while getopts "p:d:t:m:b:s:h" opt; do
    case $opt in
        p) model_path="$OPTARG" ;;
        d) device_id="$OPTARG" ;;
        t) tp="$OPTARG" ;;
        m) model_name="$OPTARG" ;;
        b) backend="$OPTARG" ;;
        s) service_port="$OPTARG" ;;
        n) parallel="$OPTARG" ;;
        r) run_locally=true ;;
        a) IFS=',' read -ra custom_apis <<< "$OPTARG" ;;  # 解析多个 API 地址
        h) usage; exit 0 ;;
        *) echo "Invalid option: -$OPTARG"; usage; exit 1 ;;
    esac
done

# Check if all required variables are set
if [[ -z "$model_name" || -z "$backend" ]]; then
    echo "Error: Missing required arguments."
    usage
    exit 1
fi

# Check if custom APIs are provided
if [[ "$run_locally" == true && ${#custom_apis[@]} -eq 0 ]]; then
    echo "Error: -a requires at least one API endpoint."
    usage
    exit 1
fi

if [[ "$run_locally" == true ]]; then
    IFS=',' read -ra model_names <<< "$model_name"  # 分割 model_name
else
    model_names=("$model_name")  # 作为单个字符串处理
fi

model_list_args=""
for model in "${model_names[@]}"; do
    model_list_args+=" $model"
done

# Variables
LMDEPLOY_IMAGE_TAG="harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:0.5.3-4a8b6d06"
CONTAINER_NAME="LLMjudge-test"
service_name="0.0.0.0"

declare -g lmdeploy_pid=""
# Function to start the model server
function open_model_server() {
    echo "Starting model server..."
    if [[ $backend == "lmdeploy" ]]; then
        if command -v lmdeploy >/dev/null 2>&1; then
            lmdeploy serve api_server \
                ${model_path} \
                --server-name ${service_name} \
                --server-port ${service_port} \
                --tp ${tp} \
                --max-batch-size 512 \
                --cache-max-entry-count 0.9 \
                --session-len 8192 &
            lmdeploy_pid=$!
        else
            docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id \
                --privileged --shm-size=10g --ipc=host \
                -v ${model_path}:/workspace/models/${model_path} \
                -p ${service_port}:${service_port} \
                --name=${CONTAINER_NAME} ${LMDEPLOY_IMAGE_TAG} \
                lmdeploy serve api_server \
                    /workspace/models/${model_path} \
                    --server-name ${service_name} \
                    --server-port ${service_port} \
                    --tp ${tp} \
                    --max-batch-size 512 \
                    --cache-max-entry-count 0.9 \
                    --session-len 8192
        fi
    else
        echo "Invalid backend specified: $backend"
        exit 1
    fi

    sleep 5m
    echo "Waiting for server to start..."
}

# Function to stop the model server
function close_model_server() {
    echo "Stopping model server..."
    if command -v lmdeploy >/dev/null 2>&1; then
	    # pkill -f "lmdeploy serve api_server"
	    kill -9 "$lmdeploy_pid"
    else
    	container_id=$(docker ps -a | grep ${CONTAINER_NAME} | awk '{print $1 }')
    	if [[ -n "$container_id" ]]; then
        	docker stop ${container_id}
        	docker rm ${container_id}
    	else
        	echo "No container found with name ${CONTAINER_NAME}"
    	fi
    fi
}

# Run the pairwise result generation script
cd ../..
pip install -e ".[model_worker,llm_judge]"
pip install --upgrade openai
cd fastchat/llm_judge

export OPENAI_API_KEY=sk-2c27e4a764b24dd79d83e6c6e65362fa
export OPENAI_API_BASE=https://api.deepseek.com

if [[ "$run_locally" == false ]]; then
    # Start model server
    open_model_server

    python3 gen_pairwise_result.py \
        --model-list $model_list_args \
        --parallel $parallel \
        --openai-api-base http://$service_name:$service_port \
        --local-api \
        --mode pairwise-all

    # Stop model server
    close_model_server
else
    api_base_args=""
    for api in "${custom_apis[@]}"; do
        api_base_args+=" $api"
    done

    python3 gen_pairwise_result.py \
        —-model-list $model_list_args \
        —-parallel $parallel \
        —-openai-api-base $api_base_args \
        —-local-api \
        —-mode pairwise-all
fi
