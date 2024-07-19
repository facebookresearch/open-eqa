export OPENAI_API_KEY=<YOUR_API_KEY> 
export OPENAI_AZURE_DEPLOYMENT=1 
python open-eqa-predict.py --method gpt4o --force --verbose --dry-run
python evaluate-predictions.py data/predictions/gpt4o-dry_run-predictions.json --force --verbose --dry-run
