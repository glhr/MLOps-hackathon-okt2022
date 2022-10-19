gcloud builds submit . -t gcr.io/rational-energy-366007/data-scraper-image
gcloud run deploy reddit-fetch-service     --image=gcr.io/rational-energy-366007/data-scraper-image     --no-allow-unauthenticated     --service-account=fake-person-53@rational-energy-366007.iam.gserviceaccount.com     --memory=1Gi     --region=us-central1     --project=rational-energy-366007
export api=$(gcloud run services describe reddit-fetch-service --region us-central1 --format 'value(status.url)')
export token=$(gcloud auth print-identity-token)
curl $api/reddit -XPOST -H 'content-type: application/json' -H "Authorization: Bearer $token" -d '{"dest": "rational-energy-366007.socials.reddit-posts"}'
