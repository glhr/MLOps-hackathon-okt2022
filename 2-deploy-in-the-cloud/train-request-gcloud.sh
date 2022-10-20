export api=$(gcloud run services describe interaction-service --region us-central1 --format 'value(status.url)')
export token=$(gcloud auth print-identity-token)

export dataset="gs://dataset-csv/output.csv"
export model="model.pt"
export train_payload=$(cat <<EOF
{
    "dataset": "$dataset",
    "model": "$model"
}
EOF
)
curl $api/train \
    -XPOST -H 'content-type: application/json' \
    -H "Authorization: Bearer $token" \
    -d "$train_payload"
