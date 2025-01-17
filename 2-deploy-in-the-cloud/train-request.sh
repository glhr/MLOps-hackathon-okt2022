export api="http://localhost:8080"
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
    -d "$train_payload"
