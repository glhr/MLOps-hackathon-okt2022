gcloud builds submit . -t gcr.io/rational-energy-366007/interaction-image
gcloud run deploy interaction-service \
    --image=gcr.io/rational-energy-366007/interaction-image \
    --no-allow-unauthenticated \
    --service-account=fake-person-53@rational-energy-366007.iam.gserviceaccount.com \
    --memory=1Gi \
    --region=us-central1 \
    --project=rational-energy-366007
export api=$(gcloud run services describe reddit-fetch-service --region us-central1 --format 'value(status.url)')
export token=$(gcloud auth print-identity-token)
