#!/usr/bin/env bash
#
# csef a.k.a Cold Start Energy Forecasting
#

################################################################################################################
## ENVIRONMENTS
################################################################################################################

export PROJ_HOME=$PWD
if [ -e "${PROJ_HOME}/gcloud-creds.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="${PROJ_HOME}/gcloud-creds.json"
fi
export GOOGLE_ML_HCDR_PROJ_ID="engaged-arcanum-217701"

################################################################################################################
## SCRIPTS
################################################################################################################

# This function define a wrapper for the csef pipeline command.
function csef-train() {

    local configs=""
    local version=""
    local sample_frac=""
    local make_submission="False"
    local save_pipeline="True"
    local seed=40

    while [ "$1" != "" ]; do
        PARAM=`echo $1 | awk -F= '{print $1}'`
        VALUE=`echo $1 | awk -F= '{print $2}'`
        case ${PARAM} in
            -v | --version)
                version=${VALUE}
                ;;
            -sf | --sample)
                sample_frac=${VALUE}
                ;;
            -ms | --make-submission)
                make_submission=${VALUE}
                ;;
            -dp | --dump-pipeline)
                save_pipeline=${VALUE}
                ;;
            -s | --seed)
                seed=${VALUE}
                ;;
            *)
                configs=${PARAM}
                ;;
        esac
        shift
    done

    # Start the pipeline
    csef -rl run \
        -cf ${configs} \
        -v ${version} \
        -sf ${sample_frac} \
        -ms \
        -dp \
        -s ${seed}
}

# This script print the log of a session
function csef-download-raw-data() {
    ./bin/download-dataset.sh
}

# This script print the log of a session
function csef-log() {
    csef -rl log -sid $1
}

# This script download all submission files
function csef-sync-submissions() {
    csef sync-submissions
}

# This script download all trained model files
function csef-sync-training() {
    csef sync-training -sid $1
}

# This script show all status of training process
# How to:
#   print all: csef-training-status
#   print all of config: csef-training-status -cf pipeline-configs/truocpham/catboost-gpu-baseline.yml
#   print with limit results: csef-training-status --limit 10
function csef-training-status() {
    csef -rl training-status $@
}

# This script do ensemble top submissions based on AUC threshold
function csef-ensemble-submission() {
    csef -rl ensemble-submission $@
}

# This script clean the unfinished results
# NOTE: really dangerous and can delete running result.
function csef-clean-training-records() {
    csef -rl clean-training-records
}

function csef-submit-google-ml-job() {
    local MAIN_TRAINER_MODULE="csef.google_ml"
    local PACKAGE_STAGING_PATH="gs://ml-csef-bucket"
    local now=$(date +"%Y%m%d_%H%M%S")
    local JOB_NAME="kaggle_csef_$now"
    local REGION="us-east1"

    cp -rf ./pipeline-configs ./csef
    cp -rf ./gcloud-creds.json ./csef
    cp -rf ./config ./csef

    # Package first
    python setup.py sdist

    rm -rf ./csef/pipeline-configs
    rm -rf ./csef/gcloud-creds.json
    rm -rf ./csef/config

    # Owner of the job
    local owner="$(git config user.name)"

    # Start submit the job
    gcloud --project=${GOOGLE_ML_HCDR_PROJ_ID} ml-engine jobs submit training ${JOB_NAME} \
        --staging-bucket ${PACKAGE_STAGING_PATH} \
        --module-name ${MAIN_TRAINER_MODULE} \
        --packages ./dist/csef-1.0.0.tar.gz,./third-party-libs/xam-0.0.1.dev0.tar.gz \
        --region ${REGION} \
        --config ./google-ml-configs/baseline-config.yml \
        --runtime-version 1.8 \
        -- \
        -v $1 \
        -t $2 \
        -cf $3 \
        -ex gzip \
        -dp \
        -ms \
        --owner gcloud-ml-${owner}
}

function csef-cancel-google-ml-job() {
    # Start submit the job
    gcloud --project=${GOOGLE_ML_HCDR_PROJ_ID} ml-engine jobs cancel $1
}

# This script install needed packages for the current environment
function csef-install() {
    pip install -e .
}

function csef-serve-lab() {
    open http://localhost:8888/lab
    nohup jupyter lab --ip="0.0.0.0" &
}

# This script helps to run jupyter notebook
function csef-serve-notebook() {
    jupyter notebook
}

# This script will help to run any command
# Usage: csef-nohup -cmd=[command here]
function csef-nohup() {
    local command=""

    while [ "$1" != "" ]; do
        PARAM=`echo $1 | awk -F= '{print $1}'`
        VALUE=`echo $1 | awk -F= '{print $2}'`
        case ${PARAM} in
            -cmd | --command)
                command=${VALUE}
                ;;
        esac
        shift
    done

    if [ "${command}" != "" ]; then
        # Start background command
        echo "Run background for command: ${command}"
        nohup ${command} > "/tmp/${command}.log" &
    else
        echo "ERROR: There are no command to run. Example: csef-nohup -cmd=csef-serve-lab"
    fi
}

# This script helps to list out all of background running scripts under current ENV
function csef-nohup-list() {
    ps aux | grep $VIRTUAL_ENV
}
