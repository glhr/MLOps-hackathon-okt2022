#export api=$(gcloud run services describe interaction-service --region us-central1 --format 'value(status.url)')
#export token=$(gcloud auth print-identity-token)

export api="http://localhost:8080"
#export token=$(gcloud auth print-identity-token)
export dataset="gs://dataset-csv/output.csv"
export model="model.pt"
export predict_payload=$(cat <<EOF
{
    "sample": "$1",
    "model": "$model"
}
EOF
)
curl $api/predict \
    -XPOST -H 'content-type: application/json' \
    -d "$predict_payload"
